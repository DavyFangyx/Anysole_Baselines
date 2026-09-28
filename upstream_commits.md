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
