# MotionPRO 数据制备（InsoleAdapted）

MotionPRO-InsoleAdapted 的输入链以 `shared/facts` 为唯一来源，由私有 adapter 生成：

```bash
# C1 审计（代表 session，禁止训练）：
python -m AnysoleWorkspace.tool.adapters.MotionPRO.adapter --session S5091 --audit
# C2 全量（用户确认 C1 后）：
python -m AnysoleWorkspace.tool.adapters.MotionPRO.adapter --split train
```

输出在 `../../AnysoleWorkspace/model_inputs/MotionPRO/adapter_v1/cam<id>/<date>/<subject>/<session_id>/`：

```text
pressure.npz      # dual-sole raster (T,160,120)，固定左右块冻结口径
contact.npy       # 私有四档 soft-f6 脚损失权重 (T,10)，仅 6/7 列非零
frame_id.npy      # shared frame id
artifact.json     # 声明 contact 语义 + 转换口径
```

`contact.npy` 不是公共接触 GT；四个软值为 `{0.05, 0.30, 0.70, 0.95}`，
其余八列为 0。visual 特征/bbox/关键点仍由 `lib/util/` 的生成脚本补齐
（bbox → image feature → kps，见 MotionPRO README 第 2 节）。

## Split：只读验证

canonical split 位于 `../../AnysoleWorkspace/protocol/splits/default/splits.csv`
（92/12/36，val=test，已冻结）。MotionPRO 不得自行划分；`make_splits.py`
只做只读验证（计数、去重、受试者隔离、session 存在性），不写任何文件：

```bash
python data_prep/make_splits.py
```

不再支持 `--ood/--exclude/--force` 等划分参数，也不再支持自行生成 splits.csv。
