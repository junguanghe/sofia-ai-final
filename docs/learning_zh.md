# KV cache 学习笔记

建议先看一遍 `README.md` 跑通命令，再按下面的顺序读代码。每次只弄懂一个问题。

## 1. 这个模型到底学什么？

数据是莎士比亚文本，每个不同字符对应一个整数，共 65 种。这不是单词 tokenizer：一个空格、一个换行也各算一个 token。

假设一段输入是 `ROMEO`，训练目标是后移一个字符后的 `OMEO:`。
模型在每个位置预测下一个字符。`train.py` 用 cross-entropy 衡量预测分布和真实字符的差距，随后 `backward()` 和 AdamW 更新参数。

训练时一次计算整段的所有位置。因果掩码阻止某个位置偷看右边的目标字符。
验证集不参与梯度更新。我们的验证 loss 是固定抽取的 20 个 batch 的平均值，不是整个验证集逐字符遍历的精确平均值。

## 2. 先追踪一次普通 forward

打开 `model.py` 的 `TinyGPT.forward`，按顺序看：

1. `tokens` 的形状是 `[B, T]`。训练时通常为 `[8, 128]`。
2. token embedding 把每个字符变成 128 维向量。
3. position embedding 告诉模型这个字符在窗口中的位置。
4. 两个 `Block` 分别做 attention 和逐位置的前馈计算，两部分都有残差连接。
5. `lm_head` 把隐藏向量变成 65 个 logits。logits 是未归一化的分数，softmax 后才是概率。

本项目采用课程中的 pre-LayerNorm 结构：先归一化，再做 attention 或 FFN。
我们没有实现原始 Transformer 论文中的完整 encoder-decoder 或 cross-attention。

## 3. Q、K、V 是什么？

每个 head 有三个不同的线性变换，分别生成 Q、K、V。它们的权重来自训练。

对一个 head，`d_head = 128 / 4 = 32`。普通 forward 的形状是：

| 张量 | 形状 | 用途 |
| --- | --- | --- |
| Q | `[B, T, 32]` | 每个位置用什么特征寻找上下文 |
| K | `[B, T, 32]` | 上下文位置用什么特征被匹配 |
| V | `[B, T, 32]` | 匹配后聚合的信息 |
| Q @ K.transpose(-2, -1) | `[B, T, T]` | 每个 query 与每个 key 的匹配分数 |

除以 `sqrt(32)` 后，先应用因果掩码，再对 key 这一维 softmax，最后乘 V。
四个 head 的输出拼接回 128 维，经投影后接残差连接。

## 4. 为什么历史 K、V 不变？

在第一层，历史字符的输入由字符和固定位置编号决定，不会因为右侧增加字符而变化。
因果 attention 又保证历史位置只能读取自己及左侧的位置。前馈层和 LayerNorm 都逐位置计算。
所以第一层历史位置的输出不变，第二层对应的历史输入也不变。逐层推下去，每层的历史 K、V 都可以保存。

这个解释依赖：权重不变、评估模式关闭 dropout、位置编号不变，以及因果掩码正确。
本实现只在 `model.eval()` 时允许缓存。`torch.inference_mode()` 另外关闭梯度记录；它不会自动关闭 dropout。

## 5. 缓存路径比原始路径多了什么？

先看 `Head.forward` 中的 `past`：

```python
q, k, v = self.query(x), self.key(x), self.value(x)
if past is not None:
    k = torch.cat((past[0], k), dim=1)
    v = torch.cat((past[1], v), dim=1)
```

decode 时 `x` 只包含一个新位置。Q 的形状为 `[B, 1, 32]`，拼接后 K、V 为 `[B, 历史长度+1, 32]`。
attention 分数因此只有一行。历史位置的 attention 和 FFN 不再重算。
但新 query 仍然要访问所有历史 K、V，所以单步计算量仍随历史长度增长。

`cache[layer][head]` 保存一个 `(K, V)` 元组。缓存作为参数传入、作为返回值传出。
模型实例里没有持久的请求缓存，下一次 `generate()` 从 `cache = None` 开始。
`torch.cat` 会分配并复制张量，代码简单但仍有开销；本项目没有实现预分配缓存。

## 6. 两个最容易写错的细节

**位置编号。** 如果已有 20 个字符，新字符的位置编号应为 20。不能因为本次只输入一个字符，就又使用位置 0。
`TinyGPT.forward` 根据缓存长度生成 `positions`。

**掩码行号。** 新 query 的绝对位置从 `offset` 开始。
`self.causal[offset:offset + T, :K_length]` 取的是正确的历史行。
这既支持单字符 decode，也支持一次追加多个字符。直接对一个 `1 x K_length` 的矩阵取普通下三角，会错误地只允许看第一个 key。

本项目限制 `prompt长度 + 生成长度 <= 128`。课程代码超长时会裁剪输入并重置窗口位置编号，旧缓存此时不能直接复用。
我们明确拒绝超长输入，没有实现 sliding-window cache。

## 7. 为什么生成 64 个字符只有 63 次 decode？

prefill 输入整个 prompt，最后位置的 logits 已经预测了第一个输出字符。
把第一个输出字符送回模型，才得到第二个字符的预测。如此反复，64 个输出只需要后续 63 次 forward。

因此最终缓存长度是 `prompt长度 + 63`，最后生成的字符还没有被送回模型。
当 prompt 为 64 时，保存 127 个位置，缓存为 254 KiB；若实际送入第 128 个位置，才达到容量上限对应的 256 KiB。

## 8. 正确性检查应该看什么？

`test_cache.py` 比较完整序列 forward 和增量 forward 的所有位置 logits。
它还覆盖逐 token 输入、多 token 分块、batch=2、缓存长度、因果性、新请求隔离和上下文边界。
有 checkpoint 时测试该权重，否则测试一个小型随机模型。正式结果必须对应训练完成后的 checkpoint。

浮点运算顺序不同可能产生很小误差，因此 logits 使用 `atol=1e-5, rtol=1e-4`。
greedy 生成另检查字符序列完全相同。若两个候选分数极接近，小浮点误差也可能改变 argmax；所以文本相同不能替代 logits 检查。

## 9. 如何读懂性能实验？

打开 `benchmark.py` 的 `replay`：两条路径读取同样的验证集字符续写。
这是 teacher-forced replay，用来固定每一步输入，避免随机采样使工作负载发生变化。
它模拟生成所需的 forward 调用，但不包含采样、拼接输出、分词、加载文件等时间。
因此 `total_ms` 表示 prefill + decode 的模型执行时间，不是一个完整应用的响应时间。

每个配置预热两次。正式运行五轮，每轮使用一个不同的验证集片段，并随机打乱六个配置的执行次序。
结果报告中位数和最小到最大范围。五轮只是小规模描述性实验，不支持稳定的 p99 或置信区间结论。
测速时不同时训练；CPU 线程数固定为 4。普通桌面背景负载和 CPU 温度仍可能影响结果。

两条路径都只对最后位置计算最终 vocabulary logits，避免把一个无关的输出层优化混进缓存对照。

## 10. 内存和时间怎样估算？

普通多头 attention 的 K、V 字节数：

```text
2 * layers * batch * cached_length * width * bytes_per_element
```

前面的 2 代表 K 和 V。所有 head 的宽度加起来等于模型宽度，因此不能再额外乘 head 数。
本模型 FP32 每个数 4 字节，每增加一个缓存位置增加 2 KiB。
这是持久 K、V 张量大小，不包括权重、Python 对象、临时 attention 张量和 cat 时的短暂额外内存。

单层、忽略常数时，无缓存处理长度 T 的历史大致需要 `O(T*d^2 + T^2*d)`；
缓存后一次新 token 的主要计算约为 `O(d^2 + T*d)`。这不是固定倍数的加速预测，真实耗时还受矩阵大小、CPU 库和 Python 调用影响。

## 阅读路线（约 45–60 分钟）

- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)：选择第 3.2 节 attention 和第 3.5 节位置编码。理解公式与位置的作用。
- [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)：选择第 2.1–2.2 节的推理过程，以及第 3 节开头的内存问题。论文的 PagedAttention 是进一步的内存管理方法，本实验只实现普通连续张量缓存。
- [Karpathy 课程代码](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py)：只对照 attention、Block 和 generate。其余部分按需回看。

## 自测与动手练习

1. 不看代码，画出 batch=1、历史长度=20 时一个 head 的 Q、K、V 形状。
2. 为什么不需要保存历史 Q？为什么深层的缓存也能复用？
3. 把 `sample.py` 的生成长度改为 20，并比较缓存开关的 greedy 输出。
4. 在单独的临时实验里把位置 offset 改错，预测哪项测试会失败，然后恢复。
5. 算出 batch=2、缓存长度=80 时的 K、V 大小，再用 `cache_bytes` 验证。
6. 不看讲稿，用两分钟解释为什么缓存加速不等于生成质量提升。

建议答辩前用自己的话回答以上问题，再复跑一次测试和测速。学习笔记只能辅助理解，实际掌握程度要靠你走读和动手确认。
