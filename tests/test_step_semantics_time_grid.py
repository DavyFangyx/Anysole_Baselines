from pathlib import Path
import sys,json,tempfile,unittest,types,os
import numpy as np
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
stage = ROOT/'Baselines'
sys.path.insert(0,str(ROOT));import Baselines.utils
Baselines.utils.__path__.insert(0,str(stage/'utils'))
sys.path[:0]=[str(stage/'Step2Motion/src'),str(ROOT/'Baselines/Step2Motion/src')]
from model import PriorTransformer
from skeleton_layout import GAIT_JOINT_NAMES,metric_joint_indices
from inference_windows import inference_starts,inference_window
from dataset import MotionDataset
from metrics import test as infer,get_metrics
from losses import PoseLoss,TranslationLoss
from config import PredictionMode
from Baselines.utils import gt_loading as gt
gt.REPO_ROOT = ROOT
from Baselines.utils.prediction_grid import bvh_prediction_on_grid
from bvh_export import write_bvh

torch.set_num_threads(1)
class IdentityNormalizer:
 def normalize_poses(self,x):return x
 def normalize_insole(self,x):return x
 def denormalize_poses(self,x):return x
class Oracle(torch.nn.Module):
 def forward(self,x,distances,t,control,c,c_mask,**kwargs):return c[...,3:],[]
class TransOracle(torch.nn.Module):
 def forward(self,x,c):return c[...,:3]
class Diffusion:
 T=1
 def q_posterior_mean_from_x_0(self,x,p,t):return p

def fake_dataset(n):
 d=MotionDataset.__new__(MotionDataset);x=torch.arange(n*69).reshape(n,69).float()/1000
 d.poses=[x];d.insole=[x.clone()];d.distances=torch.ones(1,22);d.clip_frame_ids=[np.arange(7,7+n)];d.clip_valid=[np.ones(n,bool)];return d

def prior(layout='semantic_v1'):
 return PriorTransformer(100,69,32,1,4,64,0.,10,pose_layout=layout)

class Repairs(unittest.TestCase):
 def test_anatomical_routing(self):
  m=prior();captured={}
  def hook(k):
   def f(module,args):captured[k]=args[0][0,0].detach().numpy()[::3].astype(int).tolist()
   return f
  for key in ['left_leg','right_leg','body']:getattr(m,key+'_emb').register_forward_pre_hook(hook(key))
  x=torch.arange(1,23).repeat_interleave(3).float()[None,None,:].repeat(1,3,1)
  y,_=m(x,torch.ones(1,22),torch.ones(1,dtype=torch.long),None,None,None)
  self.assertEqual(captured['left_leg'],[15,16,17,18]);self.assertEqual(captured['right_leg'],[19,20,21,22]);self.assertEqual(y.shape,x.shape)
  self.assertTrue(torch.equal(x.index_select(-1,m.pose_group_order).index_select(-1,m.pose_ungroup_order),x))
  y.square().mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
 def test_checkpoint_layout_guard(self):
  semantic=prior();legacy=prior('legacy_positional')
  with self.assertRaisesRegex(RuntimeError,'Legacy positional'):semantic.load_state_dict(legacy.state_dict())
  semantic.load_state_dict(prior().state_dict())
  bad=semantic.state_dict();bad['pose_group_order']=bad['pose_group_order'].roll(1)
  with self.assertRaisesRegex(RuntimeError,'differs'):semantic.load_state_dict(bad)
  legacy.load_state_dict(prior('legacy_positional').state_dict())
 def test_semantic_metrics(self):
  y=torch.zeros(3,69);p=y.clone();p[:,3+3*16]=1 # LeftFoot, full-skeleton index 17
  m=get_metrics(IdentityNormalizer(),y,p,40,GAIT_JOINT_NAMES)
  self.assertAlmostEqual(m[3][0],1/8);self.assertEqual(len(m[0][2]),3)
  p.zero_();p[:,3]=1 # Spine
  self.assertEqual(get_metrics(IdentityNormalizer(),y,p,40,GAIT_JOINT_NAMES)[3][0],0)
 def test_actual_inference_boundaries(self):
  for n in [1,49,50,51,99,100,101,149,150,151,258,450,1000]:
   with self.subTest(n=n):
    d=fake_dataset(n)
    result=infer(d,IdentityNormalizer(),Diffusion(),torch.nn.Identity(),Oracle(),TransOracle(),PoseLoss('cpu'),TranslationLoss(),dict(input_T=100,output_dim=69,prediction_mode=PredictionMode.X0),1.,torch.device('cpu'),False,False)
    self.assertEqual(len(result[3]),n);self.assertTrue(torch.equal(result[2],result[3]));self.assertEqual(result[0],0);self.assertEqual(result[1],0)
    np.testing.assert_array_equal(d.inference_frame_ids,np.arange(7,7+n));self.assertTrue(d.inference_valid.all())
 def test_gaps_fail_instead_of_inpainting_wrong_frames(self):
  d=fake_dataset(101);d.clip_frame_ids[0][50:]+=1
  with self.assertRaisesRegex(ValueError,'split gaps'):inference_window(d,0,100)
 def test_real_36_tail_coverage(self):
  d=torch.load(ROOT/'AnysoleWorkspace/model_inputs/Step2Motion/adapter_v1/gait/gait_test.pt',map_location='cpu',weights_only=False)
  total=0
  for clip in range(d.n_clips()):
   item=d.isolate_clip(clip);n=len(item.poses[0]);emitted=[]
   for i,start in enumerate(inference_starts(n,100)):
    _,_,_,ids,mask,real=inference_window(item,start,100);emitted.extend(ids[0 if i==0 else 50:real])
   np.testing.assert_array_equal(emitted,d.clip_frame_ids[clip]);total+=len(emitted)
  self.assertEqual(d.n_clips(),36);self.assertEqual(total,sum(len(x) for x in d.poses))
 def test_shared_time_all_channels(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'frames.npz';q=np.array([.375006,.400006,.425006]);np.savez(path,frame_id=np.arange(3),mocap_time_s=q,visual_time_s=q-.3,valid=np.ones(3),fake=np.zeros(3))
   row=dict(session_id='fixture',n_frames='3',target_fps='40',visual_start_s='0.075006',offset_s='0',shared_frames_path=str(path),smpl_path='/tmp/unused.npz',bvh_path='/tmp/unused.bvh')
   for _ in range(2):np.testing.assert_array_equal(gt.session_time_grid(row),q)
   with patch.object(gt,'load_motion',return_value=dict(joints=np.zeros((3,24,3)),names=['x']*24)) as fn:gt.protocol_gt(row,'smpl24');np.testing.assert_array_equal(fn.call_args.kwargs['query_t'],q)
   with patch.object(gt,'load_rotations',return_value={}) as fn:gt.protocol_gt_rotations(row,'smpl24');np.testing.assert_array_equal(fn.call_args.kwargs['query_t'],q)
   with patch.object(gt,'load_surface',return_value={}) as fn:gt.protocol_gt_surface(row,'smpl24');np.testing.assert_array_equal(fn.call_args.kwargs['query_t'],q)
 def test_training_rejects_actual_overlap(self):
  import train
  from config import load_config
  cfg=load_config(str(ROOT/'results/baselines/Step2Motion/checkpoints/gait_model/copy_config.json'))
  cfg['model_dir']='/tmp/step_semantics_should_not_write'
  cfg['val_data']=cfg['train_data']  # Explicit overlap, independent of the current split.
  with patch('torch.cuda.is_available',return_value=False), patch.object(
   train,'copy_config',side_effect=AssertionError('Overlap guard must stop before training setup')):
   with self.assertRaisesRegex(ValueError,'datasets overlap'):train.main(cfg,True)
 def test_old_clock_metrics_cache_cannot_be_silently_reused(self):
  from Baselines.utils.evaluate import write_comparison
  from Baselines.utils.solver import COMPARISON_SCHEMA_V2
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);output=root/'metrics/test_comparison.json';output.parent.mkdir();output.write_text(json.dumps(dict(schema_version=COMPARISON_SCHEMA_V2)))
   entry=dict(name='fixture',protocol='smpl24',prediction_root=str(root/'predictions/eval_motion'))
   args=(entry,'test',Path('/tmp/split.csv'),Path('/tmp/manifest.csv'),[],{}, {},set(),set(),'GT',False,False)
   with self.assertRaisesRegex(FileExistsError,'old/undeclared clock'):write_comparison(*args)
   entry['protocol']='pressure'
   self.assertEqual(write_comparison(*args),output)
 def test_real_140_clock(self):
  ws=ROOT/'AnysoleWorkspace';rows=[]
  for group in ['train','val','test']:rows+=gt.read_manifest(ws/'protocol/manifests/session_manifest.csv',group,ws/'protocol/splits/default/splits.csv',evaluable_only=False)
  self.assertEqual(len(rows),140)
  for row in rows:
   frames=gt.shared_session_frames(row);np.testing.assert_array_equal(gt.session_time_grid(row),frames['mocap_time_s'])
 def test_bvh_frame_map_and_rotation_alignment(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'sample.bvh';names=['Hips','LeftFoot'];parents=np.array([-1,0]);offsets=np.array([[0.,0,0],[0,-1,0]])
   quats=np.zeros((3,2,4));quats[...,0]=1;root=np.array([[0.,0,0],[1,0,0],[2,0,0]])
   write_bvh(p,names,parents,offsets,quats,root,.025)
   # The loader expects native BVH-23, so use the real skeleton for the roundtrip.
   d=torch.load(ROOT/'AnysoleWorkspace/model_inputs/Step2Motion/adapter_v1/gait/gait_test.pt',map_location='cpu',weights_only=False)
   quats=np.zeros((3,23,4));quats[...,0]=1
   write_bvh(p,list(d.joint_names),d.parents.numpy(),d.offsets[0].numpy(),quats,root,.025)
   with self.assertRaisesRegex(ValueError,'no canonical frame map'):bvh_prediction_on_grid(p,dict(session_id='S',n_frames=8))
   np.savez(p.with_suffix('.frames.npz'),frame_indices=np.array([2,3,4]),valid_mask=np.array([True,False,True]),session_id=np.asarray('S'))
   points,mask,names,rots=bvh_prediction_on_grid(p,dict(session_id='S',n_frames=8))
   self.assertEqual(np.flatnonzero(mask).tolist(),[2,4]);np.testing.assert_allclose(points[2:5,0],root,atol=1e-6);self.assertEqual(len(rots),8)
   np.savez(p.with_suffix('.frames.npz'),frame_indices=np.array([2,2,4]),valid_mask=np.ones(3,bool),session_id=np.asarray('S'))
   with self.assertRaisesRegex(ValueError,'duplicated'):bvh_prediction_on_grid(p,dict(session_id='S',n_frames=8))

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Repairs))
 sys.exit(0 if result.wasSuccessful() else 1)
