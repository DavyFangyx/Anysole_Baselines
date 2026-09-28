# 上游 provenance

本仓库首提交 `111220c` = 四个上游仓库工作树的纯净快照（生成时逐一验证 `git status --porcelain` 为 0 行）。

> 快照遵循上游仓库自身 `.gitignore` 规则；因此上游跟踪但被其自身忽略的文件不在快照内，仅一例：`Step2Motion/data/dancing/dance_{train,val,test}.pt`（被上游 `data/.gitignore` 的 `dancing/*.pt` 排除），见偏离 #1。

| 模型目录 | 上游 commit | 上游 remote | 快照日期 |
| --- | --- | --- | --- |
| `MotionPRO/` | `325f48550eb82a64e7630fcbf54404e0a387e66b` | https://github.com/wjrzm/MotionPRO.git | 2026-09-28 |
| `Step2Motion/` | `a7bdc4033eac42791071b6d09b6c0e4508aac772` | https://github.com/JLPM22/Step2Motion.git | 2026-09-28 |
| `VP-MoCap/` | `b88fce85121087ab9a4f6d4da0bb79633fb3673b` | https://github.com/wjrzm/VP-MoCap.git | 2026-09-28 |
| `pressure_tookit/` | `305969f086e4edb7316ca6a6b1b794ac97f9ed7d` | https://github.com/haolyuan/pressure_tookit.git | 2026-09-28 |

审计坐标：对任何模型的语义改动，以 `git diff <首提交> -- <模型>/` 为对照；登记表见 `语义改动对照.md`。

上游 .git 目录已存档至（本仓库之外）：

```text
/data/fangyuxuan/projects/baselines_upstream_archive/git/{MotionPRO,Step2Motion,VP-MoCap,pressure_tookit}.git
```

## 本地偏离 #0（数据层，快照时即生效）

| 项 | 内容 |
| --- | --- |
| 对象 | `Step2Motion/data/UnderPressure/`（上游跟踪的 10 个 zip，704M） |
| 处置 | 移出本仓库，存档至 `/data/fangyuxuan/projects/baselines_upstream_archive/data/UnderPressure/` |
| 原因 | 上游打包数据集（与本项目 gait 数据无关）；压缩包存在大小写两套重复（`UnderPressure.zip.00X` 与 `underpressure.zip.00X`） |
| 恢复方式 | 从存档路径移回即可；如后续做上游复现对照需要原始文件 |

## 本地偏离 #1（数据层，2026-09-28 用户裁定）

| 项 | 内容 |
| --- | --- |
| 对象 | `Step2Motion/data/dancing/` 上游 demo 数据（0.bvh / 0.json / 0.txt / dance_train.pt 16.6M / dance_val.pt 4.7M / dance_test.pt 4.7M） |
| 处置 | 整体移出本仓库。`0.*` 三件于提交 `09e07ad` 删除；`dance_*.pt` 因上游 `data/.gitignore` 的 `dancing/*.pt` 规则在首提交 `111220c` 时即未入库（用户随后删除工作树中的 `data/` 目录） |
| 原因 | 上游 demo 数据集/检查点，与本项目 gait 数据无关 |
| 恢复方式 | `git --git-dir=/data/fangyuxuan/projects/baselines_upstream_archive/git/Step2Motion.git show a7bdc40:data/dancing/<file>` |

## 本地偏离 #2（数据协议层，2026-09-28 用户裁定 D0–D7）

第一步「原始数据与 splits 适配」的交付契约决策。裁决与审计记录在主仓库
`z_note/重构执行/10_原始数据适配_决策与执行记录.md`；本表为归档索引。

| D | 交付契约 | 裁定 | 执行 commit |
|---|---|---|---|
| D0 | 底座（原始数据四部分 + splits + 协议口径）已明确；splits 104/36/36 | 登记 | — |
| D1 | SMPL-24 neutral + 10 betas，raw 原样交付（y-up/m，D4 修正后口径） | 选项 A | — |
| D2 | BVH-23 / Y-X-Z / cm / 120Hz / 72ch 原样交付；保留 output_dim 69（原生 xsens 分支背书，66/69 降级说明项） | 选项 A | e94f1ff |
| D3 | 40Hz 为交付帧率；PoseTransOpt 重力 dt→1/fps，其余基线帧基仅登记 | 选项 A | bf26a98 |
| D4 | raw 按原样交付（SMPL y-up/m、BVH cm）；z-up/m 转换归各 adapter | 修正后成立 | — |
| D5 | valid/fake 双时钟作为 raw 元数据统一交付，各 adapter 自处理并登记 | 选项 A | — |
| D6 | 相机标定统一交付 = protocol/calibration/<date>.json（外部只读）；adapter 换算原生三件套；focal 按日期取 cam3 真实 fx（join_manifest camera 块，yaml 值仅默认回退） | 选项 A | 5e31c52 |
| D7 | 触觉 raw 协议 = 左/右 PressureWasher CSV（48 数值列 = 4×12/脚、t_us、valid_mask/fake、final_fake_marked 来源）；量程（满量程/增益）无记录，契约事实 | 选项 A（不改码） | — |

数据层偏离 #0/#1（上游数据集移出）保持不变。

## 本地偏离 #3（数据层，2026-09-29 用户裁定归档处置）

| 项 | 内容 |
| --- | --- |
| 对象 1 | 主库冗余副本 `Baselines_Backup copy/`（实测 997M） |
| 处置 1 | 压缩 `archive/Baselines_Backup_copy_20260929.tar.gz`（1.03G，636 条目校验）存入本仓库 archive/；**原件已删除**（冗余副本，无代码引用） |
| 恢复方式 1 | `tar -xzf archive/Baselines_Backup_copy_20260929.tar.gz` |
| 对象 2 | `Baselines_old/`（326M，参考树） |
| 处置 2 | 压缩 `archive/Baselines_old_20260929.tar.gz`（164M，764 条目校验）；**原件保留**——`AnysoleWorkspace/tool/adapters/{MotionPRO/run_visual_chain.py, Step2Motion/tests.py, mmvp_series/depth/rgb2depth.py}` 仍引用 |
| 对象 3 | `Baselines/utils` 符号链接 → `../Baselines_old/utils`（第 4 步 utils 迁移被叫停后的导入桥） |
| 处置 3 | 保持符号链接，`.gitignore` 登记 `/utils` 与 `/archive/`（均不入库） |
