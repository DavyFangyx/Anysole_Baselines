# 01 · Baseline 复现说明（总索引）

生成日期：2026-09-29。性质：D→T→M→E 四阶段复现项目的**一页总索引**；一切裁定细节以 `Baselines/决策/` 五份档案为准。
数据来源：`决策/10`–`14` 五份档案 + `模型输入与监督信号.md` + 两库 `git log`（本文所有 commit 号均已用 `git log --oneline` 实测）。

## §1 一句话总述

四基线（MotionPRO / Step2Motion / FPP-Net / PoseTransOpt / pressure_tookit，来自 `Baselines/` 的四个上游仓库：MotionPRO、Step2Motion、VP-MoCap、pressure_tookit）+ 主模型 AnySole 的可审计复现已走完 **D（数据协议）→ T（中间产物与模型接入）→ M（执行优化）**，**E（评估）已启动**：U7/U9/U10 已裁定（2026-10-01）、E6 已执行；正式结果生产仍按 `决策/04_E_评估收尾.md` §6 阻塞中；两库分工：**Baselines 库 = 原生代码 + 决策档案 + 偏离记录**，**主库 = 数据生产（`AnysoleWorkspace/tool/adapters/`）+ 公共评估协议**；**正式结果一律用原生口径生成（M 加速开关关闭）**。

## §2 底座（已裁定，不再讨论）

| 项 | 内容 |
| --- | --- |
| raw·相机 | 每日期 `protocol/calibration/<date>.json`（K/R/t/畸变/分辨率；1624×1240、方格 6.0mm）；focal 按日期取 cam3 真实 fx（六日期 1328.6336–1409.4525） |
| raw·触觉 | 左/右 PressureWasher CSV：52 列 = frame_idx/t_us/valid_mask/fake + 48 数值列（4×12/脚）；**量程（满量程/增益）无记录**（契约事实） |
| raw·视觉 | cam3 RGB @ 40Hz（交付帧率 40Hz 为 D3 裁定） |
| raw·动捕 | BVH-23 / Y-X-Z / cm / 120Hz / 72ch；SMPL-24 neutral + 10 betas，**y-up / m**（raw 原样交付，z-up 转换归各 adapter） |
| splits | `protocol/splits/default/splits.csv` 实况 **104/36/36**（105 行含表头；val=test 为采集协议事实） |
| 评估协议 | 公共 evaluator + R_Test1/2/3/4 原生协议（缺失能力标 `—`，不写 0） |

细节见 `决策/01_D_数据协议.md`。

## §3 系列决策总表

### 3.1 D 表：数据协议层（细节见 `决策/01_D_数据协议.md`）

| 编号 | 模型/对象 | 决策与改动 | commit | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| D0 | 底座与 splits | 底座 = raw 四部分 + 公共评估接入；splits 104/36/36 登记 | 主库 d74201e（首档）、2c6bf8b（补录） | splits.csv 逐行实况核对；旧任务书 92/12/36 归 U10 | 已裁定，仅登记 |
| D1 | SMPL 交付契约 | 统一 SMPL-24 neutral + 10 betas；raw 原样交付 | 主库 d74201e、14689db（§3 D1 补充） | poses (1620,72) 轴角 / betas (10,) / neutral / markers 53 label；消费侧 `image_pressure.py:39-86`、`body_models.py:143-163` | 已裁定；smpl.npy 生产链归 C2（已完成） |
| D2 | BVH 交付契约 | 保留 BVH-23 / 69 维；交付 = raw BVH 原样（Y-X-Z / cm / 120Hz / 72ch） | Baselines e94f1ff（config_gait 移植）、主库 d74201e、2c6bf8b | 23 关节与 SRC_JOINTS 逐名一致；Frame Time 0.00833333；cm 双口径并账（27.9 前臂 / 46.554 全量最大） | 已裁定 |
| D3 | 40Hz 策略 | 40Hz 为交付帧率；PoseTransOpt 重力 dt 0.033→1/fps + MMVP.yaml 显式 `fps: 40.0` | Baselines bf26a98、主库 d74201e | 原生 `initial_trans.py:49` 实测 dt=0.033s（30fps 硬编码）；MotionPRO 20 帧窗 @40Hz=0.5s | 已执行；余 4 模型仅登记 |
| D4 | 坐标/单位边界 | raw 层按原样（SMPL y-up/m、BVH cm）；z-up/m 转换归 adapter | 主库 d74201e、2c6bf8b | raw trans 三轴范围 Y[1.179,1.189] 为身高轴 → 实测 y-up；npz 自述 "+Y up, +Z forward" | 已裁定（y-up 契约修正成立） |
| D5 | valid/fake 与双时钟 | 作为 raw 元数据统一交付，各 adapter 自处理并登记 | 主库 d74201e | `frames.npz` 四列实测（501 帧 valid=1 / 11 帧=0 / fake 全 0）；四 adapter fake 处理已登记 | 已裁定，无需改码 |
| D6 | 相机标定交付格式 | 统一每日期 calibration json；MMVP.yaml 换新采集口径；focal 按日期取 cam3 真实 fx | Baselines 5e31c52、主库 09f57f7、54e36cb | 六日期 cam3 fx 实测（1394 = 0804 值四舍五入）；S5011 冒烟 fx=1393.9955459261741 | 已裁定已执行 |
| D7 | 触觉 raw 协议 | 交付 = 左右 CSV（48 数值列 = 4×12/脚）+ 量程/t_us/fake 来源；量程缺失为契约事实，归一化由消费方处理 | 主库 2c6bf8b（补 D7）、54e36cb（量程声明） | 52 列实测；跨 session 3.2× 漂移探究 = 右脚 cell 35 单通道故障（占超量程质量 76.5%），bulk 稳定 | 已裁定；4 条契约事实待用户补录 |
| D0–D7 归档 | 偏离 #2（数据协议层） | 归档为偏离条目 | Baselines b41d506、be369d0 | 档案即登记表（原登记表 09-29 移出） | 已归档 |

### 3.2 T 表：中间产物与模型接入（细节见 `决策/02_T_方案.md`）

| 编号 | 模型/对象 | 决策与改动 | commit | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| T1 | 主库生产（六模型输入） | 谁消费谁生产：各 adapter 自产中间产物；MMVP 三模型数据端合并（`mmvp_series/`）；MotionPRO 虚拟毯 + smpl.npy 链；AnySole labels + HRNet 全量 | 主库 bceb8ee、27a9819、bc0834a、3f08603、a829970、9120884、671453a、0686ccd（T1-01…T1-11） | MotionPRO 140 session 六件套 + 视觉链 140/140；HRNet 140/140（52,612 帧）；AnySole 10 方法 × 140 = 1400 npz、793 项测试全过；SAM3.1 重建后 index 52,612 行 time_error null=0 | 完成（T1 验收 gate 汇总表 §18 待填） |
| T2-01 | MotionPRO | 登记 #1–#10（M1 contact 列/关节分离、M2 四档 soft-f6、M5 窗口索引）+ 虚拟毯 + smpl.npy 新链读取 + 训练收尾评估模块移植；消费端断言 (T,160,120)→(T,320,120) | Baselines 2ae7572 | 8 个可移植测试全过 + 整轮训练 exit 0（epoch 0 Train 2134.89 / MPJPE 1.64）；36 session 评估落盘 | 完成 |
| T2-02 | Step2Motion | 登记 #1–#12；模型侧 config（output_dim 69、input_dim 50/38）+ imu_available 开关 + normalizer 来源 + test.py 导出修正（zyx）；T1 侧原生 .pt 重导出 | Baselines 0a1bb34 | import 24/24、normalizer 逐位一致、BVH 往返 4e-8、1-epoch 双变体训练 | 完成 |
| T2-03 | FPP-Net | 登记 #1–#8：press2Cont th=0.5（正式回填）、顶点 BCE 192 维、insole 来源改 `mmvp_series` 自产树、frame-id join、连续 bce 导出 | Baselines fc1010f | 7 测试（双环境）+ one-batch exit 0（首 batch 0.5756） | 完成 |
| T2-04 | PoseTransOpt | 登记 #1–#9；#5/#6 由 D 系列承载（逐日期 fx、dt→1/fps）；CLIFF 单人、β 均值、keep-mask、分段 savgol、shape 10 基、输出契约 | Baselines 71d8466 | 26 测试 + S5011 GPU 冒烟 EXIT=0 | 完成 |
| T2-05 | pressure_tookit | 登记 #1–#5、#10–#17（13 条）；契约强化 4 条 + fitting bug 修复（reset_params/NaN）+ 输入来源替换；#18 halpemap 26→25 维持现状仅登记 | Baselines 3db7efd | 13 测试 + one-frame 两阶段（init_shape 28s / init_pose 35s）EXIT=0 | 完成 |

### 3.3 M 表：执行优化（细节见 `决策/03_M_方案与清单.md`）

| 编号 | 模型/对象 | 决策与改动 | commit | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| M1–M4 | pressure_tookit | M1 GPU ICP **不转正**（正式口径 = CPU，平方距离缺陷记录不修）；M2 画布 640,576 + 内参缩放**转正**；M3 maxiters=101 定案；M4 `run_full_mmvp` 迁入 canonical + 调度接入 | Baselines 7706cc6、主库 536dfd5 | S10101 5 帧链：GPU 仅 1.74× 且等价不成立（transl 1.75mm / loss +2.8~4.3%）；640 比 native 快 2.56×；30/50 次欠收敛、100→300 平坦盆地 | 完成（遗留已登记） |
| M12 | FPP-Net | 启用 batch 32 + workers 4 + batch 路径关 cuDNN TF32（`--batch_size 1` 可回原生逐帧口径） | Baselines cedd921、主库 bf26da7 | 关 TF32 后批/逐帧 max ≤2.4e-7（12,566 帧全 split）；端到端 7.0×（88.6→12.7s）；逐字节一致不可达（浮点归约序 1–2 ULP，已证伪） | 完成 |
| M8 | Step2Motion | 训练早停（epochs_pose 100 / epochs_trans 200 → 收敛曲线 + 早停机制） | — | 收敛曲线 + 最优 epoch 对照待出；ckpt 目录 09-28 23:51 被更新 = 实验在跑 | 进行中（最终 ckpt 待 M8 结论） |
| M10 | MotionPRO | batch_size 16→64 复核（显存 vs 刻意） | — | 显存/速度实测 + 训练曲线对照待出 | 进行中（后台任务） |
| M13 | 通用（四学习型模型） | AMP 扫描（fp16 / bf16 对照，无代码改动） | — | fp16 四模型全不达标（MotionPRO val MPJPE +16%、AnySole loss +55%、Step2Motion pose val +10.1%）；bf16 仅 AnySole 全链达标（1.68×、下游 VT2M 292.5 vs 297.3；MotionPRO bf16 0.63× 无加速） | 终表待出 / 待拍板 |
| A1–A5 | 既有 M 资产（汇总） | A1 GPU ICP 分支、A2 18→2.8 min/帧提速系列、A3 FPP 训练早停、A4 断点续跑、A5 可视化开关（默认关） | Baselines 3db7efd、fc1010f、7706cc6、71d8466 | T2 移植时随行成 M 条目；A1/A2 已拍板（不转正 / 转正） | 已具备 |

### 3.4 E 表：评估收尾（细节见 `决策/04_E_评估收尾.md`）

| 编号 | 模型/对象 | 决策与改动 | commit | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| E1 | 同体系比较（主线） | 六模型正式结果全过公共 evaluator → 统一指标表 + 每模型协议说明行 | — | 两套协议：SMPL-24（AnySole/MotionPRO/PoseTransOpt/pressure_toolkit）、BVH-23（Step2Motion）、V2T（FPP-Net） | 待 T 后启动 |
| E2 | 能力门控矩阵 | 26 键 × 六模型主矩阵 + 三条硬规则（能力≠数组存在 / provenance 不门控 / AND 三件套）；漂移点 a–k 待修订 | 主库 9de5d30（E 方案入档） | §3.3 主矩阵；逐 `—` 依据逐项可追溯 | 矩阵已成稿；E2-1 定版待裁、E2-2~6 待办 |
| E3 | 公共 contact GT 口径 | 五问待裁：Q1 GT 源（建议 press2Cont 顶点级）/ Q2 是否公共 / Q3-R35 三问 / Q4 th=0.5 / Q5 `contact_f1` 是否恢复 | — | 真实消费面 = FPP-Net 一个；磁盘四处口径互斥（f6_soft / press2Cont / motion_f6 / th=0.7） | 待拍板（`—` 影响面仅 FPP 行 + 展示层） |
| E4 | 指标口径 | foot sliding 单一公式（`metrics.py:364`，接触判据 0.3 m/s 只看 GT）+ 按协议关节集；U7 PVE 启用 vs 禁用；跨 session 尺度对齐 | — | 主模型与基线同一函数对象（`solver.py:32` 直接 import）；两套关节集 ankle↔Foot / foot↔ToeBase | E4-1 确认现状；U7 已裁定（启用）；E4-3 登记 |
| E5 | 交付 gate 与阻塞清单 | A/B 可跑、C 基本不可跑、D–G 依赖；U9 门槛已裁定（删除定义）；推荐导出顺序（C3 先行） | — | B 阶段 101+61+34 全过；`fpp_checkpoint=false`（最长链）；C1 实为 36/36 从头跑；唯一齐备块 = T2M | 阻塞中（细节见下） |
| E6 | 三 SMPL-24 基线（MotionPRO / VP-MoCap / pressure_toolkit） | 导出侧补齐 `surface_source` / `shape_source` / `public_surface_metrics` 元数据（pressure_toolkit 已有；MotionPRO = `model_prediction`、VP-MoCap = `method_optimization`，与 `models_modes.yaml` sources 一致；注册表无需改）；`evaluate.py` 闸门加固待 H13 后随行（遗留） | 主库 cb51240、Baselines 6c38948 | B 阶段三测试 34/101/61 复跑全过 + 三写入点写-读往返冒烟（负例含缺省 true / false 关闭） | 完成 |
| U7 | 关键裁定（PVE 口径） | **已裁定（2026-10-01）：拍 A（启用）** + 表注"拟合族/预测族不作排名" + 三基线 provenance 统一（E6 条目）；备选"降为明细"被否决；旧任务书禁令已废止删除 | — | 详见 `决策/04_E_评估收尾.md` §5.2、§5.4 | 已裁定（启用 + 表注 + E6） |
| U9 | 关键裁定（完整性门槛） | **已裁定（2026-10-01）：删除「可发布」全部定义**——不预设完整性门槛、随做随发布；原三方案作废 | — | 详见 `决策/04_E_评估收尾.md` §6.6 | 已裁定（删除定义） |
| 其余待拍板 | E3-Q1–Q5 / E4-1 / E2-1 等 | §8 待拍板清单为唯一汇总（含四份归档全部挂账） | — | `决策/04_E_评估收尾.md` §8 | 待裁 |

### 3.5 R 表：风险裁定（细节见 `决策/05_R1-R3_风险裁定.md`）

| 编号 | 模型/对象 | 决策与改动 | commit | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R1 | 虚拟毯入口空间尺度 + 世界锚点 | 现状入口 96×96 足印像素面积**精确减半**、长轴 40% 源行永不进入口；锚点偏移半个走廊（大量空毯）；建议方案 D（入口 192×96） | — | 足印 44.237 vs 原生 88.474 px²；入口像素 41.667 vs 20.833mm；**53.8% 帧整幅毯为空、30/140 session 全空**；方案 D 前向 +62.3% | 待裁定（用户指示先不处理；正式编号 R1/R2 映射见 §六） |
| R2 | smpl.npy GT 监督链 | **几何口径通过、时间对齐不通过**：轴/单位/尺度/R-t 全对，但图像↔mocap 配对偏 +9.5 帧（237.5ms） | — | 静止帧 4.5px 底线 vs lag0 逐关节中位 39.9px；亚帧极小点 +9.25~+9.75 帧；独立第二 2D 源（YOLOX）IoU 峰 lag +10 帧（0.604→0.717）；全库残差 −17…+26 帧 | 待裁定 |
| R3 | 故障格 cell 35 传导 | 污染成立且为三独立通道：**权重膨胀 ≫ 输入饱和 ≫ 二值假接触**；附带发现侧车 GT 用已退役 th=0.7 + 38 session 旧权重 | — | 1.48% 接触对假阴性（256,041/17,258,724）；8/484 维 20.05% 帧 sigmoid>0.99；假接触 160 帧（0.313%）封闭在 3/96 顶点 | 待裁定（方案 A/A′/B/C 已备） |

## §4 六模型状态总表

（内容取自 `模型输入与监督信号.md` + `决策/11`、`13`、`14`。**输出协议**：`smpl24` = SMPL-24；`bvh23` = BVH-23；`V2T` = 触觉→运动九项族。）

| 模型 | 类型 | 输入与生产者 | 输出协议 | 监督 | 适配状态 | 正式结果 | 关键风险 / 裁定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MotionPRO | 学习型·回归（FRAPPE） | 图像特征 (T,2048) + 压力毯 (T,320,120)；`tool/adapters/MotionPRO/`（adapter + run_visual_chain） | smpl24：(N,85) = beta(10)+theta(72)+transl(3) | **唯一有独立 GT**：`smpl.npy` / `keypoints.npy` / `contact.npy` | T1 ✓ 0686ccd（140 session 六件套）+ T2 ✓ 2ae7572 | 36/36 npz，mtime 09-24（是否属新链待 M10 核） | 锚点空毯 53.8%（R1）、入口丢行（R1）、时钟 +9.5 帧（R2）；ckpt 属 09-22 旧链需重训 |
| Step2Motion | 学习型·条件扩散 | 条件 (T,50)（16 压/足 + IMU）+ x0 (T,66→69)；`build_gait.py` | bvh23：(T,66) → 反归一化 → `_gen.bvh` | 训练 `.pt` 的 poses 字段自身（无接触监督） | T1 ✓ bceb8ee（原生 .pt 重导出）+ T2 ✓ 0a1bb34 | 36/36 目录存在，12 个有 plain `_gen.bvh`，仅 S14063 有 c0/c1 变体，其余 23 个无产物 | variant 规则待定；M8 早停实验在跑；评估 legs/toes 指标名实不符（审计 S2） |
| FPP-Net | 学习型·时序网络 | (5,26,3) 关键点 + 31×11×2 鞋垫；`tool/adapters/FPP-Net/` | V2T：(484) 压力重建 + (192) 顶点接触概率 | press2Cont 二值（网格级 + 顶点级），加载时由输入派生、**无独立 GT 文件** | T2 ✓ fc1010f + M12 ✓ cedd921 / 主库 bf26da7 | **1/36**（S14011 smoke） | **生产 ckpt 全库缺失 → 必须重训（最长链 C3）**；侧车 GT th=0.7 错配；R3 故障格三通道 |
| PoseTransOpt | 优化型·逐帧优化 | CLIFF 先验 + 2D 关键点 + FPP 4-flag + 模板深度（RGB 仅可视化）；`mmvp_series/` | smpl24：`opt_result.pth` {pose (T,24,3,3), beta (T,10), trans (T,3)} | 无训练监督（四项损失目标全部由运行时输入派生） | T2 ✓ 71d8466 + D6 逐日期 fx（09f57f7）+ D3 dt（bf26a98） | 归 VP-MoCap 行 **1/36** | 依赖 C3（FPP `pred_contact_smpl`）；M5–M7 待启动；`run_full_mmvp --gpu 0` exit 1 待修 |
| pressure_tookit | 优化型·SMPLify 式 | RGB-D + insole 31×11 + 2D 关键点 + CLIFF + 地面/标定；`mmvp_series/pressure_tookit/` | smpl24：地面系 SMPL npz + OBJ 网格 | 无训练监督（insole 9 区域二值标签 + GMM 先验） | T2 ✓ 3db7efd + M1–M4 ✓ 7706cc6 / 主库 536dfd5 | **0/36** | C1 实为 36/36 从头跑；depth 58 session 未接 adapter；floor 数据全缺；U7（PVE）待裁 |
| AnySole（主模型） | 学习型·多模态主线 | 四模式 VT2M/V2M/T2M/V2T，由 config_id 展开；`anysole/`（主库，不在 Baselines 库） | smpl24（V2T 行记 `pressure`）：`predictions/eval_motion/<sid>_<config>.npz` | 鞋垫标签 `contact_method ∈ {joint_and, motion_f6, f6_soft, pressure_f6}`；接触指标为诊断键 | T1 ✓ bc0834a（labels 1400 npz）+ 9120884（HRNet 140/140）；**HMR 已舍弃** | 11 基座 ckpt 齐；12 基座 × 144 npz 已齐（09-30 产出，血缘与口径待核） | 不在 `models_modes.yaml`（矩阵唯一例外） |

## §5 复现路径（通用五步）

1. **底座已定**（§2）：raw 四部分 + splits 104/36/36 + 每日期标定 json + 评估协议，不再讨论。
2. **中间产物由各 adapter 自产**（T1，"谁消费谁生产"）：每输入带 producer / artifact.json / 源 hash / 消费端读取测试。
3. **原生代码改动 = 登记条目 + commit**：`git diff 111220c -- <模型>` 恰好等于该模型全部登记条目（T + M）。
4. **M 加速开关默认关**：正式复现不用 M 路径；M 证据（开销表 + 精度 diff 表）只进对照，不进正式结果。
5. **E 正式评估走原生口径**：统一导出 → 公共 evaluator → 指标表/展示端；M 开关打开的产物不进 E 正式表。

## §6 已知偏差（一行一条；括号内为证据位置）

1. 深度派生世界 **~11% 度量亏空**（≈ −6.5% 垂直；头顶 1498–1526mm vs 真值 1622mm）——维持现状仅登记（`02_T_方案.md` §17，主库 671453a）。
2. 时钟：共享事实层视觉↔动捕配对偏 **+9.5 帧（237.5ms）**，全库残差 −17…+26 帧；S1 扫描中位 −5.8 帧（`05_R1-R3` §三）。
3. 虚拟毯锚点偏移半个走廊：**53.8% 帧整幅毯为空、30/140 session 全空**（`05_R1-R3` §二）。
4. 入口 96×96：**足印像素面积精确减半**（44.237 vs 88.474 px²）+ 长轴 40% 源行永不进入口（`05_R1-R3` §二）。
5. 故障格 cell 35 三通道污染：**1.48% 假阴性 / 20.05% 帧饱和 / 0.313% 假接触（封闭在 3/96 顶点）**（`05_R1-R3` §四）。
6. FPP 侧车顶点 GT 实为**已退役 th=0.7**（非 0.5）+ 38 个 session 用旧权重（S8×9/S12×11/S13×18）（`05_R1-R3` §四 4-5）。
7. V3 接触判据一致度 **0.66–0.80**（LA 0.716 / LF 0.664 / RA 0.800 / RF 0.707）——已降为诊断、不作硬 gate（`02_T_方案` §4.3/§13 T1-11；`05_R1-R3` §二）。
8. legs/toes 指标名实不符：Step2Motion `metrics.py:320-352` 关节索引按上游骨架硬编码（"MPJPE Legs" 实测为脊柱/颈/头/肩等）（`z_note/评估/评估配置说明书.md` §5）。

## §7 文件地图

| 文件 | 角色 |
| --- | --- |
| `决策/01_D_数据协议.md` | D0–D7 裁定 + 审计实测 + commit 对照 + D7 遗留（量程/漂移探究） |
| `决策/02_T_方案.md` | T1-0 结构裁定 + T1/T2 方案 + T1 执行记录（T1-01…T1-11）+ §17 深度口径 |
| `决策/03_M_方案与清单.md` | M 准入标准 + A1–A5 已具备资产 + M1–M14 清单与执行证据 |
| `决策/04_E_评估收尾.md` | E1–E5（E2 矩阵 / E3 接触 GT / E4 口径 / E5 阻塞）+ §8 待拍板唯一汇总 |
| `决策/05_R1-R3_风险裁定.md` | R1–R3 证据与方案 + S1/S2 时钟全库扫描 + 交叉影响与复现脚本 |
| `模型输入与监督信号.md` | 唯一输入清单：每模型原生协议 / 本地适配口径 / 监督与输出（T 系列验收依据） |
| `z_note/评估/评估配置说明书.md` | 当前评估配置唯一说明书（取代 `z_note/评估/` 历史文件）：§5 数据底座与已知偏差 / §6 正式结果状态与生产路径 / §8 与决策档案关系 |
| `README.md` | 仓库定位：四上游来源、审计纪律、两库契约 |
| `archive/` | `Baselines_old` 与 `Baselines_Backup_copy` 快照 tar.gz（2026-09-29） |

> **已废止**：`语义改动对照.md` 与 `upstream_commits.md` 于 2026-09-29 由用户移出（Baselines 工作树已删除，尚未提交）——**决策记录即登记表**，`决策/` 五份档案为唯一登记处；`README.md` 内相关条款待随行修订。
> `决策/` 五份档案为 2026-09-29 归档（Baselines 工作树新增，尚未提交）。
