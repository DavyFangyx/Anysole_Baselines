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

统一前提：数据生产在主库 `AnysoleWorkspace`（shared facts + 各 adapter 产物）；
**正式结果一律原生口径（M 加速开关关）**；正式评估走主库公共协议层：
导出 → `results_display/script/r_test2_compare.py` 对比（能力门控，缺能力记 `—`）。
命令均在本仓库对应模型目录下执行；每基座标注实测环境（2026-10-03 现状）。

### MotionPRO（学习型 · FRAPPE 回归）

- 环境：conda `motionpro`（py3.8，上游 README）
- 数据前置：`AnysoleWorkspace/tool/adapters/MotionPRO/` 六件套 140/140
  （虚拟毯 `pressure.npz` + `smpl.npy` 四键 + `contact.npy` **二值 {0,1}**（T2-01c）+
  keypoints/bbox/feature_hrnet + frame_id/color）
- 训练：`python -m app.train_frappe`（hydra，`config/config.yaml`；整轮 exit 0 已实测）
- 评估：`python -m app.test_frappe`（T2-01a 移植的训练收尾评估，36 session 落盘 eval_motion）
- 导出/对比：主库 `AnysoleWorkspace/tool/export_baseline_motion.py --model motionpro --split <val|test> --force` → `r_test2_compare.py`
- 注意：旧 ckpt（09-22 旧链）作废需重训；R1 锚点空毯 / R2 时钟为已登记偏差

### Step2Motion（学习型 · 条件扩散）

- 环境：touch_gait（py3.8；`src/prefer_env_site.py` 兼容层，S2M-01）
- 数据前置：`AnysoleWorkspace/tool/adapters/Step2Motion/build_gait.py`
  （R3 清洗后口径 `gait*.pt` + normalizer；`--no-imu` 变体同入口）
- 训练：`python src/train.py --config configs/config_gait.json`
  （变体：`--no-imu` / `--only_translation`；M8 早停口径：epochs_pose 100 / epochs_trans 200）
- 评估：`python src/test.py <ckpt> <skeleton.bvh> --dataset <gait_test.pt> --clip 0`（zyx 导出修正已登记）
- 对比：主库 R_Test2（BVH-23 协议行；legs/toes 指标名实不符为已登记偏差）

### FPP-Net（学习型 · V2T 时序网络）

- 环境：touch_gait（**numpy<2 铁律**：任何 build 步骤禁用 numpy≥2 的 python，P1 实测）
- 数据前置：`AnysoleWorkspace/tool/adapters/mmvp_series/fpp/build_inputs.py --force` +
  `build_metadata.py --force`（E3-A 清洗口径；pixel_weight 自愈 S7/S11）
- 训练（M12 口径，2026-10-03 实测 1410ep ≈13.6h）：

  ```bash
  CUDA_VISIBLE_DEVICES=<gpu> NVIDIA_TF32_OVERRIDE=0 python -m app.train_temporal \
    --config configs/temporalKPSMPLCont_series5_mlp.yaml --batch_size 32 --num_threads 4 --gpus "0"
  ```

- 三 split 推理：`python -m app.infer_smplcont --config configs/temporalKPSMPLCont_series5_mlp.yaml --phase <train|val|test> --batch_size 32 --num_threads 4 --no_visualization --gpus "0"`
- 导出：`AnysoleWorkspace/tool/adapters/mmvp_series/fpp/export_v2t.py --sessions <ids> --force`
  （E3-A 元数据自证：press2Cont 顶点级二值 th=0.5 + `pixel_weight_revision`）
- 对比：R_Test2 V2T 行（brief 第 7 键接触级，仅 FPP 行有值）

### PoseTransOpt（VP-MoCap · 优化型逐帧优化）

- 环境：**mmvp**（需 open3d；touch_gait 缺——B7 已登记）；VPoser 资产 `models/V02_05`
  （.gitignore 本地资产，缺失时自 `Baselines_old` 备份恢复）
- 数据前置：FPP 三 split 推理产物（`pred_contact_smpl`）+ adapter 树（join_manifest）
- 运行：`python run_full_mmvp.py --split <train|val|test|all> [--sessions S1,S2] --gpu <n> [--max-iter N] [--no-visualization] [--force]`（9/9 测试过，2026-10-03）
- 导出/对比：主库 `export_baseline_motion.py --model vp_mocap` → R_Test2（SMPL-24 行）

### pressure_tookit（优化型 · SMPLify 式）

- 环境：touch_gait（数据侧 10-03 实测；本库代码无 open3d 依赖）
- 数据前置（10-03 已生产齐）：`AnysoleWorkspace/tool/adapters/mmvp_series/pressure_tookit/build_inputs.py`
  （floor 9/9 + depthpro 52,328 帧 + insole + CLIFF；生产集合 140 内 0 缺失）
- 运行：`python run_full_mmvp.py --stage <fit|init_shape> --split <all|...> [--sessions ...] [--subject S5 --gender male]`
  （M1–M4 口径：GPU ICP 不转正（正式=CPU）、画布 640×576、maxiters 101）
- 导出/对比：主库 `export_baseline_motion.py --model pressure_toolkit` → R_Test2

### 公共评估对比（主库）

`results_display/script/r_test2_compare.py`（`--by-mode` 分块；能力门控 + 协议配对 GT；
正式结果只收原生口径产物）。指标唯一实现与能力矩阵见主库 `results_display/README_metrics.md`。
