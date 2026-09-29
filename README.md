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
