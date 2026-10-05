# Baselines（Anysole_Baselines）

四个基线模型的原生代码仓库。远程：https://github.com/DavyFangyx/Anysole_Baselines.git

| 模型目录 | 上游来源（见 `upstream_commits.md`） |
| --- | --- |
| `MotionPRO/` | wjrzm/MotionPRO @ 325f485 |
| `Step2Motion/` | JLPM22/Step2Motion @ a7bdc40 |
| `VP-MoCap/` | wjrzm/VP-MoCap @ b88fce8 |
| `pressure_tookit/` | haolyuan/pressure_tookit @ 305969f |

## 审计规则（本仓库唯一纪律）

1. 首提交 `111220c` 是四个上游工作树的纯净快照，是 `git diff` 的坐标原点。
2. **任何对原生代码的改动，必须先在 `决策/` 系列档案登记**（模型、步骤、原生行为、本地行为、原因、证据），再提交；改代码与登记录同一提交或登记先行。
3. 上游原生缺陷不静默修复；需要修正时另建 `<Model>-corrected`，或按登记条目明确偏离。
4. 数据层面的偏离记入对应 `决策/` 档案（原 `upstream_commits.md` 偏离条目已于 2026-09-29 并入并废止）。

复现总索引见 `01_Baseline复现说明.md`。

## 两库契约（与主项目仓库的关系）

- **本仓库只放**：四个模型的原生代码 + 决策档案（`决策/`）+ 复现总索引（`01_Baseline复现说明.md`）+ provenance。
- **主项目仓库放**：AnySole 主模型、数据工作区（AnysoleWorkspace）、公共评估协议、results/results_display、configs。
- 公共评估器（原 `Baselines/utils` 的公共 evaluator）属于主项目仓库的评估协议层，不在本仓库。
- 主项目仓库通过同级目录引用本仓库（两仓库需放在同一父目录下）。

## 各基座命令速查（训练 / 评估 / 运行）

统一前提：数据生产在主库 `AnysoleWorkspace`（shared facts + 各 adapter 产物，已齐）；
**正式结果一律原生口径（M 加速开关关）**；对比统一走主库 `r_test2_compare.py`（能力门控）。（GPU 编号按当时空闲卡改；每条的注释说明干什么、产物去哪）。

### MotionPRO（学习型 · FRAPPE 回归）｜touch_gait

```bash
# ① 训练（hydra 读 config/config.yaml；产物 results/baselines/MotionPRO/checkpoints/）
cd /data/fangyuxuan/projects/gait/Baselines/MotionPRO
CUDA_VISIBLE_DEVICES=4 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m app.train_frappe

# ② 评估（一体：test_frappe 直写统一契约 + 36 session 覆盖校验）
cd /data/fangyuxuan/projects/gait
CUDA_VISIBLE_DEVICES=2 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model motionpro --split test
# ③ 对比
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：旧 ckpt（09-22 旧链）作废，正式结果需 ① 重训后走 ②③④；R1 锚点空毯 / R2 时钟为已登记偏差。

### Step2Motion（学习型 · 条件扩散）｜touch_gait（py3.8）

```bash
# ① 训练（M8 口径 epochs_pose=100 / epochs_trans=200；ckpt → results/baselines/Step2Motion/checkpoints/gait_model/）
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
CUDA_VISIBLE_DEVICES=4 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python src/train.py --config configs/config_gait.json
# 无 IMU 变体（38 维输入，数据集 gait_noimu/）：
# ... src/train.py --config configs/config_gait.json --no-imu

# ② 评估（zyx 导出修正已登记；默认 dataset 取 config 内 test_data）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python src/test.py \
  /data/fangyuxuan/projects/gait/results/baselines/Step2Motion/checkpoints/gait_model

# ③ 对比（BVH-23 协议行；主库执行）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：legs/toes 指标名实不符为已登记偏差；正式结果需 ① 重训（旧 ckpt 09-22/09-28 换代）。

### FPP-Net（学习型 · V2T 时序网络）｜touch_gait（**numpy<2 铁律**）

```bash
cd /data/fangyuxuan/projects/gait/Baselines/VP-MoCap/FPP-Net

# ① 训练（M12 口径；2026-10-03 实测 1410ep ≈13.6h；ckpt → results/baselines/FPP-Net/checkpoints/tempKPSMPL_series5_mlp/） 
# 早停默认关闭（yaml=0，保持原生 1410 epoch 行为）
CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m app.train_temporal \
  --config configs/temporalKPSMPLCont_series5_mlp.yaml --batch_size 32 --early-stop-patience 200 --early-stop-min-delta 0.0005 --early-stop-lr-floor 5e-8 --num_threads 4 --gpus "0"

# ② 三 split 推理 + 导出（一体：infer → export_v2t，E3-A 元数据自证；结果 → results/baselines/FPP-Net/predictions/v2t/）
cd /data/fangyuxuan/projects/gait
CUDA_VISIBLE_DEVICES=1 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model fpp_v2t --split all

# ③ 对比（V2T 行；brief 第 7 键接触级仅 FPP 行有值）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：任何 build/导出步骤禁用 numpy≥2 的 python（P1 实测会写坏 pickle）；数据重建
= `AnysoleWorkspace/tool/adapters/mmvp_series/fpp/build_inputs.py --force` + `build_metadata.py --force`。

### PoseTransOpt（VP-MoCap · 优化型逐帧优化）｜**mmvp**（需 open3d）

```bash
cd /data/fangyuxuan/projects/gait/Baselines/VP-MoCap/PoseTransOpt

# ① 运行 + 导出（一体：run_full_mmvp → 自动导出统一 motion npz；
#    产物 → AnysoleWorkspace/work/VP-MoCap/v1/pose_optimization/ + results/baselines/VP-MoCap/predictions/eval_motion/）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model vp_mocap --split test --gpu 1
# ② 对比（SMPL-24 行）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

前置：FPP 三 split 推理产物（② 的 fpp_predictions）必须已存在；VPoser 资产
`models/V02_05`（.gitignore 本地资产，缺失时自 `Baselines_old/VP-MoCap/PoseTransOpt/models/` 恢复）。

### pressure_tookit（优化型 · SMPLify 式）｜数据侧 touch_gait · **拟合运行 mmvp**

```bash
cd /data/fangyuxuan/projects/gait/Baselines/pressure_tookit

# ① 运行 + 导出（一体：init_shape → fit → 自动导出；M 口径：ICP 固定 cpu（默认）、画布 640×576（默认）、maxiters=101（config 默认）；
#    --male/--female 必填（S14 为唯一 female；S9 整组排除不在管线）
#    槽/W 调参（2026-10-04 A/B 实测）：槽 = 并行进程数（--per-gpu），W = 每槽 kd-tree 查询线程数
#    单槽帧速实测 W=24: 23.2s / W=6: 26.8s / W=2: 38.5s —— W=24 只比 W=6 快 15%，线程开销却大 8-10 倍（OpenMP barrier 自旋空转计入负载）；4 槽×24 曾把 96 核机打到负载 ~180。
#    8 槽 × w=6（推荐）  负载 ~50-60  全量 ~1.8 天  每槽=实测配置，预测最稳
#    12 槽 × w=6        负载 ~70-85  全量 ~1.2 天  略热
#    24 槽 × w=2        负载 ~70     全量 ~1 天     最快；24 进程并发
cd /data/fangyuxuan/projects/gait
PRESSURE_KDTREE_WORKERS=6 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model pressure_toolkit --split all --gpu 3 --per-gpu 8
# ② 对比（SMPL-24 行）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

数据前置（10-03 已生产齐）：floor 9/9 + depthpro 52,328 帧（生产集合 140 内 0 缺失）；
`model_inputs/pressure_toolkit/v1/` 由主库 `adapters/mmvp_series/pressure_tookit/build_inputs.py` 生成。

### 公共评估对比（主库，全部基线通用）

```bash
cd /data/fangyuxuan/projects/gait
# 对比表（--split val|test；能力门控 + 协议配对 GT；正式结果只收原生口径产物）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

指标唯一实现与能力矩阵见主库 `results_display/README_metrics.md`。
