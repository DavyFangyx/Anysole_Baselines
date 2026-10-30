# Baselines（Anysole_Baselines）

七个基线模型的原生代码仓库。远程：https://github.com/DavyFangyx/Anysole_Baselines.git

| 模型目录 | 上游来源（见 `决策/` 档案；`upstream_commits.md` 已于 2026-09-29 废止） |
| --- | --- |
| `MotionPRO/` | wjrzm/MotionPRO @ 325f485 |
| `Step2Motion/` | JLPM22/Step2Motion @ a7bdc40 |
| `VP-MoCap/` | wjrzm/VP-MoCap @ b88fce8 |
| `pressure_tookit/` | haolyuan/pressure_tookit @ 305969f |
| `WHAM/` | 传输包 visual_baselines_20261029 包快照（见 `决策/11`） |
| `GVHMR/` | 传输包快照，固定 2024 发布版（上游 commit 44fbc1999dd57d2180e882c62c78d356d4c840d0） |
| `SAM3DB/` | 传输包快照：fast-sam-3d-body + dinov3 + mhr + sam-3d-body（见 `决策/11`） |

## 审计规则（本仓库唯一纪律）

1. 首提交 `111220c` 是四个上游工作树的纯净快照，是 `git diff` 的坐标原点。
2. **任何对原生代码的改动，必须先在 `决策/` 系列档案登记**（模型、步骤、原生行为、本地行为、原因、证据），再提交；改代码与登记录同一提交或登记先行。
3. 上游原生缺陷不静默修复；需要修正时另建 `<Model>-corrected`，或按登记条目明确偏离。
4. 数据层面的偏离记入对应 `决策/` 档案（原 `upstream_commits.md` 偏离条目已于 2026-09-29 并入并废止）。

复现总索引见 `01_Baseline复现说明.md`。

## 两库契约（与主项目仓库的关系）

- **本仓库只放**：七个模型的原生代码 + 决策档案（`决策/`）+ 复现总索引（`01_Baseline复现说明.md`）+ provenance。
- WHAM/GVHMR/SAM3DB 的目录是传输包代码快照，**只作审计与 diff 快照，不可脱离包独立运行**（import 与资产路径指向包内 workspace）；运行（推理验证）一律走包入口 `bin/vb`（`决策/11` 登记边界）。
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
CUDA_VISIBLE_DEVICES=0 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model motionpro --split test

# ③ 对比
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：旧 ckpt（09-22 旧链）作废，正式结果需 ① 重训后走 ②③④；R1 锚点空毯 / R2 时钟为已登记偏差。

### Step2Motion（学习型 · 条件扩散）｜touch_gait（py3.8）

```bash
# ① 训练（M8 口径 epochs_pose=100 / epochs_trans=200；ckpt → results/baselines/Step2Motion/checkpoints/gait_model/；
#    旧 legacy-positional ckpt 已删 2026-10-22，新训练为 semantic_v1 从零）
cd /data/fangyuxuan/projects/gait/Baselines/Step2Motion
CUDA_VISIBLE_DEVICES=4 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python src/train.py --config configs/config_gait.json
# 无 IMU 变体（38 维输入，数据集 gait_noimu/）：
# ... src/train.py --config configs/config_gait.json --no-imu

# ② 评估（zyx 导出修正已登记；默认 dataset 取 config 内 test_data；
#    待 ① 训出 semantic_v1 ckpt 后指向该目录；旧 ckpt 已删 2026-10-22）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python src/test.py \
  /data/fangyuxuan/projects/gait/results/baselines/Step2Motion/checkpoints/gait_model

# ③ 对比（BVH-23 协议行；主库执行）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：legs/toes 指标修复已随 semantic 重构落地（2026-10-22）。

### FPP-Net（学习型 · V2T 时序网络）｜touch_gait（**numpy<2 铁律**）

```bash
cd /data/fangyuxuan/projects/gait/Baselines/VP-MoCap/FPP-Net

# ① 训练（M12 口径；2026-10-03 实测 1410ep ≈13.6h；ckpt → results/baselines/FPP-Net/checkpoints/tempKPSMPL_series5_mlp/；
#    旧 ckpt 已移出（2026-10-22），本命令从零训）
# 早停默认关闭（yaml=0，保持原生 1410 epoch 行为）
CUDA_VISIBLE_DEVICES=1 NVIDIA_TF32_OVERRIDE=0 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m app.train_temporal \
  --config configs/temporalKPSMPLCont_series5_mlp.yaml --batch_size 32 --early-stop-patience 200 --early-stop-min-delta 0.0005 --early-stop-lr-floor 5e-8 --num_threads 4 --gpus "0"

# ② 三 split 推理 + 导出（一体：infer → export_v2t，E3-A 元数据自证；结果 → results/baselines/FPP-Net/predictions/v2t/）
cd /data/fangyuxuan/projects/gait
CUDA_VISIBLE_DEVICES=0 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model fpp_v2t --split all

# ③ 对比（V2T 行；brief 第 7 键接触级仅 FPP 行有值）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

注意：任何 build/导出步骤禁用 numpy≥2 的 python（P1 实测会写坏 pickle）；数据重建
= `AnysoleWorkspace/tool/adapters/mmvp_series/fpp/build_inputs.py --force` + `build_metadata.py --force`。

### PoseTransOpt（VP-MoCap 后 · 优化型逐帧优化）｜**mmvp**

```bash
cd /data/fangyuxuan/projects/gait/Baselines/VP-MoCap/PoseTransOpt

# ① 运行 + 导出（一体：run_full_mmvp → 自动导出统一 motion npz；
#    产物 → AnysoleWorkspace/work/VP-MoCap/v1/pose_optimization/ + results/baselines/VP-MoCap/predictions/eval_motion/）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model vp_mocap --split test --gpu 0

# ② 对比（SMPL-24 行）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

前置：FPP 三 split 推理产物（② 的 fpp_predictions）必须已存在；VPoser 资产
`models/V02_05`（.gitignore 本地资产，缺失时自 `Baselines_old/VP-MoCap/PoseTransOpt/models/` 恢复）。

### pressure_tookit（优化型 · SMPLify 式）｜数据侧 touch_gait · **拟合运行 mmvp**

```bash
cd /data/fangyuxuan/projects/gait/Baselines/pressure_tookit

# ① 运行 + 导出（一体：init_shape → fit → 自动导出；导出需
#    protocol/calibration/<date>.pressure_toolkit_native_to_mocap.json（目前仅 20260810 有，
#    新日期需先拟合再 native_to_mocap.py --date <日期>）。
#    M 口径：ICP 固定 cpu（默认）、画布 640×576（默认）、maxiters=101（config 默认）；
#    --male/--female 必填（S14 为唯一 female；S9 整组排除不在管线）
#    槽/W 调参（2026-10-04 A/B 实测）：槽 = 并行进程数（--per-gpu），W = 每槽 kd-tree 查询线程数
#    单槽帧速实测 W=24: 23.2s / W=6: 26.8s / W=2: 38.5s —— W=24 只比 W=6 快 15%，线程开销却大 8-10 倍（OpenMP barrier 自旋空转计入负载）；4 槽×24 曾把 96 核机打到负载 ~180。
#    8 槽 × w=6（推荐）  负载 ~50-60  全量 ~1.8 天  每槽=实测配置，预测最稳
#    12 槽 × w=6        负载 ~70-85  全量 ~1.2 天  略热
#    24 槽 × w=2        负载 ~70     全量 ~1 天    最快；24 进程并发
cd /data/fangyuxuan/projects/gait
PRESSURE_KDTREE_WORKERS=6 /data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/eval_baseline.py --model pressure_toolkit --split all --gpu 3 --per-gpu 8

# ② 单独导出
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/export_baseline_motion.py --model pressure_toolkit --split test --force

# ② 对比（SMPL-24 行）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

数据前置（10-03 已生产齐）：floor 9/9 + depthpro 52,328 帧（生产集合 140 内 0 缺失）；
`model_inputs/pressure_toolkit/v1/` 由主库 `adapters/mmvp_series/pressure_tookit/build_inputs.py` 生成。

### WHAM（学习型 · 视频姿态回归）｜推理 wham env · 调度/导出/评估 touch_gait

```bash
# ① 本地推理（RGB = raw 农场 + bbox = workspace frontend bbox.npz；产物 result.pkl → work/ 收口区）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/visual_baselines/generate.py --model wham --split test --gpu <空闲卡>

# ② 导出统一契约（预测 → results/baselines/WHAM/predictions/package_export/eval_motion/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/export_baseline_motion.py --model wham --split test --force

# ③ 评估 + 对比（SMPL-24 / V2M 行）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m Baselines.utils.evaluate --model wham --split test --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

没有落盘的中间量管线：所有中间量（bbox / 关键点 / MHR）在推理中生成、不落盘；bbox 消费项目正典
`shared/frontends/human_masks/sam31/v1/`（direct_csv_bbox）。训练不可复现（官方预训练权重，
SHA 锁定于项目资产区）。正式数字（test 36）：mpjpe 66.3 / pa 49.0 / root_ATE 2556.9（决策 11）。

### SAM3DB（学习型 · RGB→MHR→SMPL）｜推理 vb_tactile · 调度/导出/评估 touch_gait

```bash
# ① 本地推理（逐帧 Engine.infer；产物 result.pkl → work/ 收口区）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/visual_baselines/generate.py --model sam3db --split test --gpu <空闲卡>

# ② 导出统一契约（→ results/baselines/SAM3DB/predictions/package_export/eval_motion/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/export_baseline_motion.py --model sam3db --split test --force

# ③ 评估 + 对比
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m Baselines.utils.evaluate --model sam3db --split test --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

没有落盘的中间量管线：所有中间量在推理中生成、不落盘。训练不可复现（官方预训练权重）。
正式数字（test 36）：mpjpe 75.5 / pa 52.0 / root_ATE 563.1（决策 11；root_ATE 全基线最优）。

### GVHMR（学习型 · 视频姿态回归，2024 固定版）｜推理 vb_gvhmr · 调度/导出/评估 touch_gait

```bash
# ① 本地推理（复用迁移 worker run_request；产物 result.npz → work/ 收口区）
cd /data/fangyuxuan/projects/gait
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/adapters/visual_baselines/generate.py --model gvhmr --split test --gpu <空闲卡>

# ② 导出统一契约（→ results/baselines/GVHMR/predictions/package_export/eval_motion/）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python AnysoleWorkspace/tool/export_baseline_motion.py --model gvhmr --split test --force

# ③ 评估 + 对比
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python -m Baselines.utils.evaluate --model gvhmr --split test --force
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

没有落盘的中间量管线：所有中间量在推理中生成、不落盘。训练不可复现（官方预训练权重，
SHA 锁定于项目资产区）。正式数字（test 36）：mpjpe 66.6 / pa 51.4 / root_ATE 2274.3（决策 11）。

### 公共评估对比（主库，全部基线通用）

```bash
cd /data/fangyuxuan/projects/gait
# 对比表（--split val|test；能力门控 + 协议配对 GT；正式结果只收原生口径产物）
/data/fangyuxuan/miniconda3/envs/touch_gait/bin/python results_display/script/r_test2_compare.py --split test --force
```

指标唯一实现与能力矩阵见主库 `results_display/README_metrics.md`。
