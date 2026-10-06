# CL 历史观测：先记录，不改变融合策略

`class_fusion/cl_history.py` 为现有通用可靠性融合补充跨任务观测。它不增加 loss，不参与 `g_c` 或其更新幅度的计算，不改变教师、不提供额外训练样本。等权对照和动态融合默认都启用，方便比较。

## 每个增量任务开始时固定什么

首次任务没有旧类，不建立 CL 参考。对于后续任务：

1. memory 按新任务预算缩减完成后，取得当前实际保留的旧类视频，每个 ID 一次。
2. 使用与 KD 相同的上一任务最佳检查点作为教师。
3. 固定参考视频顺序、标签、教师有效样本掩码、有效旧类候选集合。
4. 在这批视频上重新测量教师可靠性，作为本任务的起点。**不直接复用上一任务用更多 memory、不同候选集合得到的历史均值。**

文件记录教师检查点文件名、SHA-256、来源 step、epoch、验证准确率；同时记录数据集、seed、视频 ID、标签、纳入掩码、候选类别、温度、最少样本数、reference_id。`reference_id` 标识这次比较的参考条件。

教师原始音视频输入和逐视频隐藏特征不额外保存；统计结束后释放临时特征。保存的是标识、两分支旧类原型分数、目标概率、预测、判别间隔、实际融合 logits 及分类统计，用于离线分析，未扩充回放训练集。

当前历史文件的 `schema_version=2`。旧版 schema 1 只有目标概率和部分汇总，没有完整分支分数、预测转换及配对正确性。普通模型检查点的 `format_version` 仍为 1，与历史数据自身的 schema 版本不同。

## 如何比较

教师与学生各自使用自己的同模态特征重新构造 LOO 原型，但使用完全相同的参考视频和旧类候选集合：

```text
R_before[c,m] = 教师在固定旧类 memory、固定旧类候选集上的平均目标概率
R_now[c,m]    = 当前学生在相同条件下的平均目标概率

signed_drop[c,m] = R_before[c,m] - R_now[c,m]
drop[c,m]        = max(signed_drop[c,m], 0)
drop_asymmetry[c] = drop[c,A] - drop[c,V]
```

`signed_drop < 0` 表示这个统计量改善；`drop_asymmetry > 0` 表示音频分支的退化量更大。这里比较的是旧类可分性变化，不把坐标旋转或单纯特征位移自动当作遗忘。

**这套旧类参考统计与驱动融合的全已见类别可靠性分开保存。** 学生中新增加的类别 logit 不进入这里的原型 softmax 分母。两个统计的用途和候选集合不同，不能直接混用。

无效状态的处理：

* 教师中的非有限/零特征样本对在任务开始时排除，并把排除名单固定；学生后来变有效也不补入。
* 类别有效样本不足，或 LOO 查询退化，标记为不可比较。
* 固定队列中有学生样本失效，或某个固定候选类原型失效，整次可靠性比较标记为不可比较；不会通过删样本/删候选类制造新的分母。
* 不可比较的下降量为 `NaN`，不是 0。分析时必须筛选 `comparable=True`。
* 运行中替换参考视频（即使属于同一类别）会报错。

## 样本级分支中间数据

`reference["teacher_reliability"]` 和 `observation["reliability"]` 具有相同的以下字段。设 N 为纳入参考的旧类视频数，K 为本任务的旧类别数，后缀 `a/v` 分别表示音频/视觉分支。

| 字段 | 形状 | 含义 |
|---|---|---|
| `sample_scores_a/v` | N × K | 归一化特征与同模态原型的余弦分数除以温度；目标原型已作 LOO |
| `sample_probability_a/v` | N | 真实类别的原型 softmax 概率 |
| `sample_prediction_a/v` | N | 原型分类预测的旧类别 ID；无法打分时为 -1 |
| `sample_top1_margin_a/v` | N | 目标分数减去最高错误类别分数 |
| `sample_log_odds_a/v` | N | 目标分数减去其他候选分数的 logsumexp；sigmoid 后等于目标概率 |
| `query_valid` | N | 两个分支都能在固定参考条件下进行 LOO 打分 |
| `accuracy_a/v` | K | 各旧类的原型预测准确率 |

分数列始终对应全局旧类 ID `0..K-1`；缺失的固定候选类列为 `-inf`。无法打分的样本分数整行为 `NaN`，不得直接对它 argmax 后当作有效预测。行顺序由 `reference["sample_row_indices"]` 给出，索引进入 `reference["sample_ids"]` 和 `reference["sample_labels"]`，与 `included_mask=True` 的原始顺序一致。

这些是同模态原型探针，不是独立训练的单模态分类器。视觉特征仍经过音频引导的注意力。保存完整分数，可以在不重新提取特征的情况下检查混淆类别、错误重叠、置信度分布及不同分数温度。

## 配对正确性和预测转换

教师和学生的 `paired_correctness` 都逐旧类保存：

* `both_correct_rate`：两分支都正确；
* `audio_only_correct_rate`：仅音频正确；
* `visual_only_correct_rate`：仅视觉正确；
* `both_wrong_rate`：两分支都错误；
* `prediction_disagreement_rate`：两个分支预测的类别不同。

前四项在有效类别上相加为 1。它们能描述两分支错误是否重叠，但不等价于因果意义的模态贡献或理论协同量。

`observation["prediction_transitions"]` 保存同一视频在教师与学生之间的变化：

```text
sample_forgotten_m = 1[教师分支 m 正确，学生分支 m 错误]
sample_recovered_m = 1[教师分支 m 错误，学生分支 m 正确]
```

每个分支均保存 N 维的 `sample_forgotten_a/v`、`sample_recovered_a/v`，以及 K 维类别均值 `forgotten_rate_a/v`、`recovered_rate_a/v`。均值的分母为该类全部可比较参考样本数，不是教师预测正确的样本数；另存 `sample_comparable` 和实际可比较 `counts`。不可比较的值为 `NaN`。

这使得“置信度下降但预测仍正确”“真正由对变错”“从错恢复为对”可以分别分析。尤其不要仅保存截断后的 `drop`，否则改善信息会丢失。上述指标全是 memory 上的原型探针变化，字段中的 forgotten 不直接代表测试集泛化遗忘。

## 额外保存的旧新混淆统计

同一次参考扫描还计算当前模型真正使用的融合 logit，逐旧类记录：

| 字段 | 含义 |
|---|---|
| `old_only_accuracy` | 只允许在旧类中选类别时，在该类参考视频上的准确率 |
| `all_seen_accuracy` | 在全部已见类别中预测时的参考准确率 |
| `old_to_new_rate` | 该旧类的参考视频被预测成新类的比例 |
| `classification_support/valid` | 实际有效分类样本数量，以及是否覆盖完整参考类别样本 |
| `confusion_matrix` | 行为旧类真实标签、列为全部已见类别预测标签的计数矩阵 |
| `sample_logits` | N × 当前模型类别数，实际使用的融合分类器输出，未作 softmax |
| `sample_prediction_old_only/all_seen` | 每个参考视频在旧类/全部已见类别中预测的类别 ID |
| `sample_valid` | 融合 logits 是否全部有限；无效视频的预测 ID 为 -1 |

教师没有新类输出，因此它的 `old_to_new_rate=0`（有效类别）。任务开始时也会观测扩展分类头后的学生，允许分析随机初始化的新类输出在训练前带来的混淆。

这些准确率是 **训练 memory 上的诊断值**，不能当作测试准确率。正式测试准确率和 F1 仍由原来的 `metrics.py` 输出。

## 观测时机和参数

默认 `--record_cl_history` 开启；`--no_record_cl_history` 完全跳过新增参考构建和观测。

后续任务在以下时机记录：

* epoch 0：教师参考建立后，学生训练开始前；
* 每 `--cl_history_interval` 个 epoch，默认 40；
* 每次融合权重即将更新时；
* 每次产生新的最佳验证检查点时；
* 最后一个 epoch，即使它不恰好落在周期上。

多个条件同时满足时只扫描一次。除 epoch 0 外，观测发生在本轮训练/验证之后、融合权重更新之前。最终观测不触发新的 gate 更新。

最佳检查点和最终检查点都包含对应 epoch 的精确观测，不会用上一次周期统计冒充当前状态。新增扫描使用 `eval/no_grad`，并恢复原模型模式以及 Python/NumPy/PyTorch 随机状态。记录会增加旧 memory 的推理、CPU 统计和文件存储开销。

## 输出位置

```text
save/metrics/RUN_NAME/cl_history/
  step_1_reference.json          # 人可读的视频清单和参考条件
  step_1_reference.pt            # 教师参考、逐视频目标概率及分类统计
  step_1_epoch_0_history.pt      # 初始化学生
  step_1_epoch_40_history.pt     # 某次学生观测
  ...
  class_history.csv             # 每次观测 × 每个旧类的汇总
```

CSV 包括 `introduced_step`、`class_age_steps`、memory/有效/实际打分数量、教师/学生可靠性、可比较标记、两种下降量、退化不对称量、教师/学生 gate、gate 版本、旧新混淆、分支原型准确率、配对正确性及预测转换率。完整的 N × K 分数矩阵留在 `.pt` 文件中，不扩展成庞大的 CSV。

普通模型检查点额外有：

```python
checkpoint["cl_history"]["reference"]
checkpoint["cl_history"]["observation"]
```

二者完整嵌入，复制检查点后仍能找到该模型对应的参考条件和观测。关闭记录或首任务时该字段为 `None`。旧 phase-9 检查点没有这个字段也能正常加载推理。`test_only` 不读取 CL 历史来做预测、不重算历史，也不修改训练时的观测文件。

读取示例（在已有 PyTorch 环境中）：

```python
import torch

checkpoint = torch.load("step_1_best_model.pt", map_location="cpu", weights_only=True)
history = checkpoint["cl_history"]
reference = history["reference"]
observation = history["observation"]
mask = observation["comparable"]
print(observation["epoch"], observation["gate_version"])
print(observation["drop_a"][mask], observation["drop_v"][mask])

# 逐视频比较：所有样本矩阵和标签使用同一行索引。
indices = reference["sample_row_indices"]
video_ids = [reference["sample_ids"][i] for i in indices]
labels = reference["sample_labels"][reference["included_mask"]]
valid = observation["prediction_transitions"]["sample_comparable"]
before = reference["teacher_reliability"]
now = observation["reliability"]
audio_probability_drop = before["sample_probability_a"] - now["sample_probability_a"]
visual_probability_drop = before["sample_probability_v"] - now["sample_probability_v"]
print(audio_probability_drop[valid], visual_probability_drop[valid])
print(observation["prediction_transitions"]["forgotten_rate_a"])
# 当前模型实际的全已见类别融合分数，与上面的行顺序一致。
fused_logits = observation["classification"]["sample_logits"]
```

## 解释边界

这些是可分性退化的代理统计，不能单独证明存在泛化遗忘。LOO 不消除 memory 过拟合，小 memory 下估计仍可能很不稳定。不同 step 的 memory 和旧类集合不同，所以应先分析每个 `reference_id` 内的教师—学生变化，不能把不同参考条件下的原始 R 直接相减或无条件累加退化量。

保存完整样本分数会增加存储。例如 N=500、K_old=90、K_seen=100 时，两路原型分数和一路融合 logits 的 float32 矩阵合计约 0.56 MB/次观测，不含其他字段和序列化开销；教师参考还会嵌入最佳/最终检查点。最佳验证结果频繁刷新时会产生更多观测文件。

当前工作只建立观测基础。历史文件没有被重新读入训练；未来如果将其中跨任务保存的信息用于新算法，应明确其保存内容与内存预算。是否把历史退化用于权重、如何兼顾弱分支恢复与旧类保持，仍需根据观测结果另行设计。
