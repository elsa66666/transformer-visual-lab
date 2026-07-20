const details = {
  tokens: { category: "INPUT", icon: "01", title: "输入 Token 序列", subtitle: "模型首先看到的不是句子，而是一串编号", description: "Tokenizer 先把文本拆成 token，再把每个 token 映射成词表里的整数 ID。Transformer 本身不直接处理汉字或单词。", formula: "text → [id₁, id₂, …, id_T]", dimensions: [["Token IDs", "T"], ["示例长度 T", "7"]], plain: "像先把一句话切成乐高积木，并给每块积木贴上编号。此时只有‘身份’，还没有可计算的语义向量。", path: ["文本", "Token IDs", "Embedding"], active: 1, example: "“小美正在学习注意力机制”会被拆成若干 token；真实拆分方式取决于模型使用的 tokenizer。" },
  tokenEmbedding: { category: "EMBEDDING", icon: "E", title: "Token Embedding", subtitle: "查表取出每个 Token 的语义向量", description: "Embedding 矩阵的每一行对应词表中的一个 token。输入 ID 不是拿去做乘法，而是用来选中那一行向量。", formula: "E_token = W_E[token_id]", dimensions: [["W_E", "|V| × d_model"], ["输出 E_token", "T × d_model"]], plain: "像拿着编号去一个超大的抽屉柜里取卡片。编号 1287 对应哪张卡片，是训练学出来的。", path: ["Token IDs", "Embedding", "X₀"], active: 1, example: "如果 d_model=768，那么每个 token 会被换成 768 个数字。相似用法的 token，向量通常也会逐渐靠近。" },
  positionEncoding: { category: "EMBEDDING", icon: "P", title: "位置编码", subtitle: "告诉模型：谁在前，谁在后", description: "Self-Attention 本身只看 token 间关系，不天然知道顺序。位置编码被加到词向量中，使后续层可以同时读取‘它是谁’和‘它在哪’。", formula: "X₀ = E_token + E_position", dimensions: [["E_position", "T × d_model"], ["输出 X₀", "T × d_model"]], plain: "同一个人坐在会议桌不同座位，角色可能不同。位置编码就是座位牌，而且必须和词义卡片做成同样长度，才能逐维相加。", path: ["Token Embedding", "+ Position", "X₀"], active: 1, example: "“狗咬人”和“人咬狗”token 类似，但位置不同。没有位置信息，模型很难区分谁做动作、谁承受动作。" },
  embedding: { category: "TENSOR", icon: "X₀", title: "初始隐藏状态 X₀", subtitle: "送入第一层 Transformer 的统一表示", description: "每一行对应一个 token，每一列对应一个隐藏特征。此后所有层都保持 d_model 这一主干宽度，方便残差相加。", formula: "X₀ ∈ ℝ^(T × d_model)", dimensions: [["序列维", "T = 7"], ["隐藏维", "d_model = 768"]], plain: "把整句话想象成一张表：一行是一个 token，一行里有 768 个数字描述它。", path: ["Embedding", "X₀", "Layer 1"], active: 1, example: "7 个 token、每个 768 维，因此 X₀ 是 7×768 的矩阵。" },
  layer: { category: "TRANSFORMER BLOCK", icon: "L1", title: "一个 Transformer 层", subtitle: "先交流，再各自加工", description: "一层通常由 Self-Attention 和 FFN 两个子层组成；每个子层外都有残差连接，并搭配 LayerNorm。多层堆叠让表示逐步变得更抽象。", formula: "X_l = Block_l(X_(l−1))", dimensions: [["输入 / 输出", "T × d_model"], ["示例层数", "N = 12"]], plain: "Attention 像开小组会：token 互相交换信息；FFN 像会后各自消化。重复很多轮，理解逐渐变深。", path: ["Attention", "Residual", "FFN", "Residual"], active: 0, example: "第一层可能主要识别局部搭配；后面的层会把语法、指代和更抽象的任务信息逐步组合起来。" },
  layernorm1: { category: "NORMALIZATION", icon: "LN", title: "LayerNorm（预归一化）", subtitle: "先把每个 Token 的数值尺度整理稳定", description: "LayerNorm 对每个 token 的隐藏维度做标准化，再用可学习参数 γ、β 调整。Pre-Norm 结构把它放在 Attention 或 FFN 之前。", formula: "LN(x) = γ ⊙ (x−μ)/√(σ²+ε) + β", dimensions: [["输入 / 输出", "T × d_model"], ["μ, σ²", "沿隐藏维计算"]], plain: "有的 token 向量数字特别大，有的特别小。LayerNorm 像统一音量，避免后续模块被某几个过大的数值带跑。", path: ["X", "LayerNorm", "Q/K/V"], active: 1, example: "它不会把不同 token 混在一起归一化，而是分别整理每个 token 自己的 768 个特征。" },
  q: { category: "ATTENTION", icon: "Q", title: "Q · Query（查询）", subtitle: "当前 Token 正在寻找什么？", description: "输入向量乘以 W_Q 得到 Q。它描述当前位置为了更新自己，想从其他 token 那里寻找哪类信息。Q 会与所有 K 做相似度计算。", formula: "Q = XW_Q", dimensions: [["X", "T × d_model"], ["W_Q", "d_model × d_k"], ["Q（每个头）", "T × d_k"]], plain: "Q 像你手里的问题清单。读到“它”时，你的问题可能是：前面哪个名词最像它指代的对象？", path: ["LayerNorm", "Q", "QKᵀ", "Softmax"], active: 1, example: "在“小美把书给小丽，因为她……”中，“她”的 Q 会倾向于寻找人物相关线索，再用 K 判断应该关注谁。" },
  wq: { category: "LEARNED WEIGHT", icon: "WQ", title: "W_Q · 查询投影矩阵", subtitle: "把通用隐藏向量变成‘提问视角’", description: "W_Q 是训练得到的参数。不同注意力头拥有各自的 W_Q，因此同一个 token 可以从语法、指代、位置等不同角度提出查询。", formula: "W_Q ← W_Q − η ∂L/∂W_Q", dimensions: [["W_Q（每头）", "d_model × d_k"], ["示例", "768 × 64"]], plain: "X 是一个人掌握的全部信息，W_Q 像一副‘我要找什么’的滤镜。训练决定滤镜最后关注哪些特征。", path: ["X", "W_Q", "Q"], active: 1, example: "某个头的 W_Q 可能让 Q 更关注‘当前词需要找主语吗’，另一个头可能关注远距离指代。" },
  k: { category: "ATTENTION", icon: "K", title: "K · Key（键）", subtitle: "每个 Token 声明：我能提供什么线索", description: "K 由输入乘以 W_K 得到。其他位置的 Q 会和这个 K 做点积；越匹配，说明这个位置越值得被关注。", formula: "K = XW_K", dimensions: [["W_K", "d_model × d_k"], ["K（每个头）", "T × d_k"]], plain: "K 像每本书贴在书脊上的关键词。Q 是检索请求，二者越匹配，这本书越可能被取下来。", path: ["LayerNorm", "K", "QKᵀ"], active: 1, example: "“小美”这个 token 的 K 可能暴露‘人物、女性、主语候选’等可供其他 token 匹配的特征。" },
  wk: { category: "LEARNED WEIGHT", icon: "WK", title: "W_K · 键投影矩阵", subtitle: "把隐藏向量变成‘可被检索的标签’", description: "W_K 决定一个 token 要暴露哪些特征给查询。它和 W_Q 协同训练，让有用的 Q–K 配对获得更高分。", formula: "K = XW_K", dimensions: [["W_K（每头）", "d_model × d_k"], ["示例", "768 × 64"]], plain: "W_Q 负责把问题翻译成检索语言，W_K 负责把资料翻译成同一种索引语言。", path: ["X", "W_K", "K"], active: 1, example: "如果训练发现主谓关系有用，相关注意力头会逐渐学会让主语的 K 与谓语位置的 Q 更匹配。" },
  v: { category: "ATTENTION", icon: "V", title: "V · Value（值）", subtitle: "真正被取走、汇入当前位置的信息", description: "V 由输入乘以 W_V 得到。Q 与 K 只负责算‘关注多少’，最终被注意力权重加权求和的是 V。", formula: "V = XW_V", dimensions: [["W_V", "d_model × d_v"], ["V（每个头）", "T × d_v"]], plain: "Q 和 K 像查目录，V 才是书里的正文。找到哪本书之后，真正读进脑子的是正文内容。", path: ["LayerNorm", "V", "Attention × V"], active: 1, example: "“她”关注“小美”后，从小美对应的 V 中取得对当前预测有用的人物信息。" },
  wv: { category: "LEARNED WEIGHT", icon: "WV", title: "W_V · 值投影矩阵", subtitle: "决定被传递的具体内容", description: "W_V 把完整隐藏状态压到一个注意力头需要传递的子空间。不同头可以传递不同类型的信息。", formula: "V = XW_V", dimensions: [["W_V（每头）", "d_model × d_v"], ["示例", "768 × 64"]], plain: "一个人知道很多事，但在不同会议上会拿出不同内容。W_V 决定这次会议真正带哪部分资料。", path: ["X", "W_V", "V"], active: 1, example: "一个头可能主要传递位置关系，另一个头可能传递实体语义；合并后共同更新 token。" },
  scores: { category: "ATTENTION SCORE", icon: "·", title: "QKᵀ · 相关性打分", subtitle: "每个查询和所有键两两比较", description: "Q 的每一行与 K 的每一行做点积，得到 T×T 的分数矩阵。第 i 行表示第 i 个 token 对各位置的原始关注分数。", formula: "S = QKᵀ", dimensions: [["Q", "T × d_k"], ["Kᵀ", "d_k × T"], ["S", "T × T"]], plain: "每个 token 都拿自己的问题，依次问全句每个 token：你和我的问题匹配吗？于是得到一张‘谁看谁’的表。", path: ["Q + K", "QKᵀ", "Scale"], active: 1, example: "若 T=7，就会得到 7×7 的矩阵；其中 S₄,₂ 表示第 4 个 token 对第 2 个 token 的原始关注分。" },
  scale: { category: "ATTENTION SCORE", icon: "√", title: "缩放 · 除以 √d_k", subtitle: "防止点积过大让 Softmax 过早饱和", description: "维度越高，随机向量点积的方差通常越大。除以 √d_k 将分数量级拉回稳定范围，使 Softmax 保留可学习的梯度。", formula: "S_scaled = QKᵀ / √d_k", dimensions: [["d_k", "64"], ["√d_k", "8"]], plain: "像把过大的音量调低。如果分数动不动就是几十，Softmax 会几乎只剩一个 1 和一堆 0，很难细调。", path: ["QKᵀ", "÷√d_k", "Mask"], active: 1, example: "本示例 d_k=64，因此把所有相关性分数除以 8，再送入遮罩和 Softmax。" },
  mask: { category: "CAUSAL MASK", icon: "M", title: "Causal Mask · 因果遮罩", subtitle: "生成时不能偷看未来 Token", description: "Decoder-only 模型把当前位置右侧的分数改成 −∞。经过 Softmax 后这些位置的概率变成 0，从而保持自回归生成。", formula: "S_ij = −∞,  if j > i", dimensions: [["Mask", "T × T 下三角"], ["未来位置权重", "0"]], plain: "像考试时用纸遮住后面的标准答案。预测第 4 个 token 时，只允许看第 1～4 个，不能看第 5 个以后。", path: ["Scale", "Mask", "Softmax"], active: 1, example: "训练时整句可以并行输入，但遮罩确保每个位置仍然只使用它当时本应看见的前文。" },
  attentionSoftmax: { category: "ATTENTION", icon: "σ", title: "Attention Softmax", subtitle: "把任意分数变成总和为 1 的关注比例", description: "Softmax 沿每一行计算，使每个 query 分配给所有 key 的权重非负且总和为 1。", formula: "A_ij = exp(S_ij) / Σ_j exp(S_ij)", dimensions: [["输入 S", "T × T"], ["输出 A", "T × T"], ["每行和", "1"]], plain: "像把 100% 的注意力预算分给前文：小美 60%，学习 25%，其他位置合计 15%。", path: ["Mask", "Softmax", "A × V"], active: 1, example: "原始分数 [2,1,0] 经 Softmax 后约为 [0.67,0.24,0.09]，仍保留强弱差异。" },
  weightedSum: { category: "ATTENTION", icon: "Σ", title: "注意力加权求和 · A×V", subtitle: "按照关注比例真正读取信息", description: "用注意力权重 A 对各位置的 V 做加权求和。结果的每一行，是当前位置从整段可见上下文中汇总的新信息。", formula: "H_head = Softmax(QKᵀ/√d_k + M)V", dimensions: [["A", "T × T"], ["V", "T × d_v"], ["H_head", "T × d_v"]], plain: "查完资料后，把重要资料多抄一点、不重要的少抄一点，最后合成一页与当前问题最相关的笔记。", path: ["Attention 权重", "× V", "Head 输出"], active: 1, example: "如果关注权重是 [0.6, 0.3, 0.1]，输出就是 0.6V₁+0.3V₂+0.1V₃。" },
  concat: { category: "MULTI-HEAD", icon: "∥", title: "Concat · 合并多个注意力头", subtitle: "把不同观察视角重新拼在一起", description: "每个头独立产生 T×d_v 的输出。将 h 个头沿特征维拼接，恢复为 T×(h·d_v) 的表示。", formula: "H = Concat(head₁,…,head_h)", dimensions: [["单头", "T × 64"], ["12 头合并", "T × 768"]], plain: "12 个小组分别研究语法、指代、位置等线索，Concat 把 12 份小组报告装订成一本。", path: ["Head 1…h", "Concat", "W_O"], active: 1, example: "12 个头×每头 64 维，拼接后正好得到 768 维。" },
  wo: { category: "LEARNED WEIGHT", icon: "WO", title: "W_O · 注意力输出投影", subtitle: "融合多头结果并投回主干维度", description: "拼接后的多头信息只是并排放在一起。W_O 负责跨头混合，并把结果投影到 d_model，才能与残差分支的 X 相加。", formula: "H_attn = Concat(head₁,…,head_h)W_O", dimensions: [["W_O", "(h·d_v) × d_model"], ["示例", "768 × 768"]], plain: "Concat 是把 12 份报告订在一起，W_O 是主编：把各组发现重新组织成一份统一结论。", path: ["Concat", "W_O", "+ Residual"], active: 1, example: "W_O 可以学习把一个头找到的主语信息与另一个头找到的时态信息组合起来。" },
  add1: { category: "RESIDUAL", icon: "+", title: "残差连接 · Attention", subtitle: "保留旧信息，再叠加新信息", description: "Attention 输出与进入子层前的主干表示直接相加。残差连接让信息和梯度拥有一条短路，便于训练深层网络。", formula: "X′ = X + Attention(LN(X))", dimensions: [["两条分支", "T × d_model"], ["输出 X′", "T × d_model"]], plain: "不是把旧笔记扔掉重写，而是在原笔记上补充会议里得到的新信息。即使新模块暂时没学好，旧信息还在。", path: ["X", "Attention", "+", "X′"], active: 2, example: "因为两边都保持 768 维，才能逐元素直接相加。" },
  layernorm2: { category: "NORMALIZATION", icon: "LN", title: "第二个 LayerNorm", subtitle: "进入 FFN 前再次稳定尺度", description: "Attention 更新后的每个 token 先经过 LayerNorm，再进入前馈网络。它与第一个 LayerNorm 参数独立。", formula: "Z = LN(X′)", dimensions: [["输入 / 输出", "T × d_model"], ["参数 γ, β", "各 d_model"]], plain: "小组会结束后，先把每个人的新笔记整理成统一格式，再让他们各自深入加工。", path: ["Attention 残差", "LayerNorm", "W_in"], active: 1, example: "虽然都叫 LayerNorm，但 Layer 1 的两个 LayerNorm，以及不同层的 LayerNorm，都有各自的 γ 和 β。" },
  win: { category: "FFN · LEARNED WEIGHT", icon: "Win", title: "W_in · FFN 输入投影", subtitle: "从 d_model 升维到更宽的 d_ff", description: "W_in 对每个 token 独立做线性变换，通常把隐藏维扩展到约 4 倍，使网络拥有更大的特征加工空间。不同 token 使用同一套 W_in。", formula: "H = X′W_in + b_in", dimensions: [["W_in", "d_model × d_ff"], ["示例", "768 × 3072"], ["H", "T × 3072"]], plain: "Attention 负责让 token 互相交换情报；W_in 像把每个人的 768 条简略笔记摊开到 3072 个工作格，方便细加工。", path: ["LayerNorm", "W_in", "GELU"], active: 1, example: "每个 token 都单独从 768 维变成 3072 维；这个步骤不会在 token 之间交换信息。" },
  gelu: { category: "FFN · ACTIVATION", icon: "G", title: "GELU · 非线性激活", subtitle: "让 FFN 不只是两个矩阵的合并", description: "如果两次线性变换之间没有激活函数，它们等价于一次线性变换。GELU 根据输入大小进行平滑门控，引入非线性表达能力。", formula: "GELU(x) ≈ 0.5x[1+tanh(√(2/π)(x+0.044715x³))]", dimensions: [["输入 / 输出", "T × d_ff"], ["操作", "逐元素"]], plain: "它像 3072 个可调节的小开关：重要特征通过得多，不合适的特征被压小，但不是生硬地一刀切。", path: ["W_in", "GELU", "W_out"], active: 1, example: "GELU 对每个数字单独处理，因此不会改变矩阵形状。现代模型也常用 SwiGLU 等变体。" },
  wout: { category: "FFN · LEARNED WEIGHT", icon: "Wout", title: "W_out · FFN 输出投影", subtitle: "把 d_ff 压回 d_model", description: "经过激活的宽特征由 W_out 混合并投回主干维度。只有恢复到 d_model，才能和 FFN 的残差分支相加。", formula: "FFN(X) = GELU(XW_in+b_in)W_out+b_out", dimensions: [["W_out", "d_ff × d_model"], ["示例", "3072 × 768"], ["输出", "T × 768"]], plain: "W_in 把材料铺开加工，W_out 把 3072 个加工结果重新压缩成 768 条有用结论，交还主干。", path: ["GELU", "W_out", "+ Residual"], active: 1, example: "W_in 和 W_out 不是互逆矩阵；它们共同训练，学会在宽空间里提取对下一层有用的特征。" },
  add2: { category: "RESIDUAL", icon: "+", title: "残差连接 · FFN", subtitle: "保留 Attention 结果，再加上 FFN 加工", description: "FFN 输出与它的输入 X′ 相加，形成本层最终输出，并送入下一层 Transformer。", formula: "X_l = X′ + FFN(LN(X′))", dimensions: [["输入 / 输出", "T × d_model"], ["层索引", "l = 1…N"]], plain: "先保留开会后的原笔记，再把个人深加工的结论补进去。这一层做的是增量修改。", path: ["X′", "FFN", "+", "X_l"], active: 2, example: "Layer 1 的 X₁ 会成为 Layer 2 的输入；形状不变，但每个数字承载的上下文含义已经更新。" },
  laterLayers: { category: "DEPTH", icon: "N", title: "多层堆叠", subtitle: "结构重复，但每层参数各自学习", description: "每层都有独立的 W_Q、W_K、W_V、W_O、W_in、W_out 与 LayerNorm 参数。相同的是计算模板，不是具体数值。", formula: "X_N = Block_N(…Block_2(Block_1(X₀)))", dimensions: [["每层主干", "T × d_model"], ["示例", "12 层"]], plain: "像连续开 12 轮讨论会。会议流程一样，但每轮的主持人和关注点不同，前一轮结论成为下一轮材料。", path: ["Layer 1", "Layer 2", "…", "Layer N"], active: 2, example: "浅层往往更容易呈现局部模式，深层整合更长程、更任务相关的信息；这不是固定硬规则。" },
  finalNorm: { category: "OUTPUT", icon: "LN", title: "Final LayerNorm", subtitle: "输出到词表前的最终归一化", description: "许多 Decoder-only 架构在最后一层后还有一次 LayerNorm，用于稳定送入语言模型头的隐藏状态。", formula: "H = LN_f(X_N)", dimensions: [["输入 / 输出", "T × d_model"], ["用于预测", "通常取最后位置"]], plain: "十二轮加工都结束后，再统一一次格式，然后交给‘词表打分器’。", path: ["Layer N", "Final LN", "LM Head"], active: 1, example: "自回归生成时，我们主要使用最后一个可见位置的隐藏向量来预测下一个 token。" },
  lmhead: { category: "OUTPUT PROJECTION", icon: "WU", title: "LM Head · W_U", subtitle: "从隐藏维映射到整个词表", description: "线性层为词表中的每个 token 产生一个 logit。很多模型会让 W_U 与输入 Embedding 权重共享，以减少参数并建立输入输出空间联系。", formula: "logits = h_last W_U", dimensions: [["h_last", "1 × d_model"], ["W_U", "d_model × |Vocab|"], ["logits", "1 × |Vocab|"]], plain: "把 768 维的最终理解交给词表里所有候选 token，让每个候选拿到一个原始分数。", path: ["Final LN", "Linear", "Logits"], active: 1, example: "若词表有 150,000 个 token，线性层一次会产生 150,000 个 logit。" },
  vocabSoftmax: { category: "OUTPUT PROBABILITY", icon: "σ", title: "词表 Softmax", subtitle: "把 Logits 变成下一个 Token 的概率", description: "对整个词表的 logits 做 Softmax，得到概率分布。实际生成还可能加入温度、Top-k 或 Top-p 采样。", formula: "P(token_i|context) = exp(z_i/τ) / Σ_j exp(z_j/τ)", dimensions: [["Logits / 概率", "|Vocab|"], ["概率总和", "1"]], plain: "所有候选词参加一次竞选。Softmax 把原始票数换成百分比；解码策略再决定直接选第一名还是按概率抽样。", path: ["Logits", "Softmax", "Sampling"], active: 1, example: "温度 τ 越低，分布越尖、输出越确定；温度越高，候选更分散、输出更有随机性。" },
  output: { category: "NEXT TOKEN", icon: "→", title: "预测下一个 Token", subtitle: "选出一个 Token，再把它接回输入继续生成", description: "选出的 token 会追加到序列末尾，模型再次执行同一流程预测下一个 token。借助 KV Cache，历史 K/V 不必每次全部重算。", formula: "y_t ~ P(· | y_<t)", dimensions: [["一次循环", "生成 1 个 token"], ["停止条件", "EOS / 长度上限"]], plain: "模型不是一次写完整句话，而是一次写一个 token：写完一个，把它当成新上下文，再继续写下一个。", path: ["概率分布", "选 Token", "追加", "再次前向"], active: 1, example: "当前选择“的”，下一轮输入就变成“…注意力机制的”，然后再预测“原理”等后续 token。" }
};

const tooltip = document.getElementById("tooltip");
const diagram = document.getElementById("transformerDiagram");
const hoverHint = document.getElementById("hoverHint");
let selectedKey = "q";
let playTimer = null;

const fallback = details.layer;
const getDetail = key => details[key] || fallback;
const formulaTex = {
  tokens: String.raw`\mathrm{text}\rightarrow[\mathrm{id}_1,\mathrm{id}_2,\ldots,\mathrm{id}_T]`,
  tokenEmbedding: String.raw`E_{\mathrm{token}}=W_E[\mathrm{token\_id}]`,
  positionEncoding: String.raw`X_0=E_{\mathrm{token}}+E_{\mathrm{position}}`,
  embedding: String.raw`X_0\in\mathbb{R}^{T\times d_{\mathrm{model}}}`,
  layer: String.raw`X_l=\mathrm{Block}_l\!\left(X_{l-1}\right)`,
  layernorm1: String.raw`\mathrm{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta`,
  q: String.raw`Q=XW_Q`,
  wq: String.raw`W_Q\leftarrow W_Q-\eta\frac{\partial L}{\partial W_Q}`,
  k: String.raw`K=XW_K`,
  wk: String.raw`K=XW_K`,
  v: String.raw`V=XW_V`,
  wv: String.raw`V=XW_V`,
  scores: String.raw`S=QK^\top`,
  scale: String.raw`S_{\mathrm{scaled}}=\frac{QK^\top}{\sqrt{d_k}}`,
  mask: String.raw`S_{ij}=-\infty,\quad \mathrm{if}\ j>i`,
  attentionSoftmax: String.raw`A_{ij}=\frac{\exp(S_{ij})}{\sum_j\exp(S_{ij})}`,
  weightedSum: String.raw`H_{\mathrm{head}}=\mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V`,
  concat: String.raw`H=\mathrm{Concat}\!\left(\mathrm{head}_1,\ldots,\mathrm{head}_h\right)`,
  wo: String.raw`H_{\mathrm{attn}}=\mathrm{Concat}\!\left(\mathrm{head}_1,\ldots,\mathrm{head}_h\right)W_O`,
  add1: String.raw`X'=X+\mathrm{Attention}\!\left(\mathrm{LN}(X)\right)`,
  layernorm2: String.raw`Z=\mathrm{LN}(X')`,
  win: String.raw`H=X'W_{\mathrm{in}}+b_{\mathrm{in}}`,
  gelu: String.raw`\mathrm{GELU}(x)\approx0.5x\left[1+\tanh\!\left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right]`,
  wout: String.raw`\mathrm{FFN}(X)=\mathrm{GELU}\!\left(XW_{\mathrm{in}}+b_{\mathrm{in}}\right)W_{\mathrm{out}}+b_{\mathrm{out}}`,
  add2: String.raw`X_l=X'+\mathrm{FFN}\!\left(\mathrm{LN}(X')\right)`,
  laterLayers: String.raw`X_N=\mathrm{Block}_N\!\left(\cdots\mathrm{Block}_2\!\left(\mathrm{Block}_1(X_0)\right)\right)`,
  finalNorm: String.raw`H=\mathrm{LN}_f(X_N)`,
  lmhead: String.raw`\mathrm{logits}=h_{\mathrm{last}}W_U`,
  vocabSoftmax: String.raw`P(\mathrm{token}_i\mid\mathrm{context})=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}`,
  output: String.raw`y_t\sim P(\cdot\mid y_{<t})`
};
const zoomGroups = {
  embedding: ["tokens", "tokenEmbedding", "positionEncoding", "embedding"],
  attention: ["layer", "layernorm1", "q", "wq", "k", "wk", "v", "wv", "scores", "scale", "mask", "attentionSoftmax", "weightedSum", "concat", "wo", "add1"],
  ffn: ["layernorm2", "win", "gelu", "wout", "add2"],
  output: ["laterLayers", "finalNorm", "lmhead", "vocabSoftmax", "output"]
};
const zoomForKey = Object.fromEntries(Object.entries(zoomGroups).flatMap(([zoom, keys]) => keys.map(key => [key, zoom])));

function setZoom(zoom) {
  const nextZoom = zoom || "attention";
  document.querySelectorAll("[data-zoom]").forEach(panel => panel.classList.toggle("active", panel.dataset.zoom === nextZoom));
  document.querySelectorAll("[data-focus]").forEach(node => node.classList.toggle("active", node.dataset.focus === nextZoom));
  const label = {
    embedding: "Embedding 局部放大图",
    attention: "Attention 局部放大图",
    ffn: "FFN 局部放大图",
    output: "输出层局部放大图"
  }[nextZoom];
  document.getElementById("viewStatus").textContent = label || "2D 教学流程图";
}

function renderDetails(key) {
  const item = getDetail(key);
  selectedKey = key;
  setZoom(zoomForKey[key]);
  document.getElementById("detailCategory").textContent = item.category;
  document.getElementById("detailIcon").textContent = item.icon;
  document.getElementById("detailTitle").textContent = item.title;
  document.getElementById("detailSubtitle").textContent = item.subtitle;
  document.getElementById("detailDescription").textContent = item.description;
  const formulaNode = document.getElementById("detailFormula");
  formulaNode.innerHTML = `\\[${formulaTex[key] || item.formula}\\]`;
  if (window.MathJax?.typesetPromise) {
    MathJax.typesetClear?.([formulaNode]);
    MathJax.typesetPromise([formulaNode]);
  }
  document.getElementById("detailDimensions").innerHTML = item.dimensions.map(([name, value]) => `<div class="dimension-row"><span>${name}</span><code>${value}</code></div>`).join("");
  document.getElementById("detailPlain").textContent = item.plain;
  document.getElementById("detailPath").innerHTML = item.path.map((label, index) => `${index ? '<span class="path-arrow">→</span>' : ''}<span class="path-chip ${index === item.active ? 'active' : ''}">${label}</span>`).join("");
  document.getElementById("detailExample").innerHTML = `<strong>例子：</strong>${item.example}`;
  document.querySelectorAll("[data-key]").forEach(node => node.classList.toggle("selected", node.dataset.key === key));
}

function positionTooltip(event) {
  const gap = 14;
  let x = event.clientX + gap;
  let y = event.clientY + gap;
  const rect = tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - gap;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - gap;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

document.querySelectorAll("[data-key]").forEach(node => {
  const key = node.dataset.key;
  const item = getDetail(key);
  node.setAttribute("aria-label", `${item.title}：${item.subtitle}`);
  if (!node.hasAttribute("tabindex") && node.tagName !== "BUTTON") node.setAttribute("tabindex", "0");
  node.addEventListener("mouseenter", event => {
    tooltip.innerHTML = `<strong>${item.title}</strong>${item.subtitle}`;
    tooltip.classList.add("show");
    hoverHint.textContent = item.title;
    positionTooltip(event);
  });
  node.addEventListener("mousemove", positionTooltip);
  node.addEventListener("mouseleave", () => { tooltip.classList.remove("show"); hoverHint.textContent = "把光标放到任意模块上"; });
  node.addEventListener("focus", () => { hoverHint.textContent = item.title; });
  node.addEventListener("blur", () => { hoverHint.textContent = "把光标放到任意模块上"; });
  node.addEventListener("click", () => renderDetails(key));
  node.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); renderDetails(key); } });
});

document.querySelectorAll("[data-view]").forEach(button => {
  button.addEventListener("click", () => {
    const view = button.dataset.view;
    document.querySelectorAll("[data-view]").forEach(item => { item.classList.toggle("active", item === button); item.setAttribute("aria-pressed", item === button ? "true" : "false"); });
    diagram.classList.toggle("view-3d", view === "3d");
    diagram.classList.toggle("view-2d", view === "2d");
    if (view === "3d") {
      document.getElementById("viewStatus").textContent = "3D 分层透视图";
    } else {
      setZoom(zoomForKey[selectedKey]);
    }
  });
});

document.querySelectorAll("[data-architecture]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-architecture]").forEach(item => item.classList.toggle("active", item === button));
    const isDecoder = button.dataset.architecture === "decoder";
    document.getElementById("diagramTitle").textContent = isDecoder ? "Decoder-only Transformer" : "Encoder–Decoder Transformer（Decoder 侧）";
    const maskNode = document.querySelector('[data-key="mask"]');
    maskNode.querySelector("b").textContent = isDecoder ? "Mask" : "Cross";
    maskNode.querySelector("span").textContent = isDecoder ? "遮住未来" : "读取编码器";
    const maskDetail = details.mask;
    if (isDecoder) {
      maskDetail.title = "Causal Mask · 因果遮罩";
      maskDetail.subtitle = "生成时不能偷看未来 Token";
    } else {
      maskDetail.title = "Cross-Attention · 交叉注意力位置";
      maskDetail.subtitle = "Decoder 用 Q 读取 Encoder 的 K/V";
    }
    if (selectedKey === "mask") renderDetails("mask");
  });
});

document.querySelectorAll(".attention-pipeline button, .ffn-pipeline button").forEach((node, index) => node.style.setProperty("--i", index));

document.getElementById("playFlow").addEventListener("click", event => {
  const button = event.currentTarget;
  const isPlaying = diagram.classList.toggle("is-playing");
  button.classList.toggle("active", isPlaying);
  button.querySelector(".play-icon").textContent = isPlaying ? "Ⅱ" : "▶";
  button.querySelector(".play-label").textContent = isPlaying ? "暂停信息流" : "播放信息流";
  clearTimeout(playTimer);
  if (isPlaying) playTimer = setTimeout(() => button.click(), 9000);
});

document.querySelectorAll(".stage-item, .mini-block[data-focus]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".stage-item").forEach(item => item.classList.toggle("active", item === button));
    if (button.dataset.focus) setZoom(button.dataset.focus);
    const target = document.querySelector(`[data-key="${button.dataset.focus}"]`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
      target.classList.remove("focus-flash");
      void target.offsetWidth;
      target.classList.add("focus-flash");
      renderDetails(button.dataset.focus);
    } else if (button.dataset.key) {
      renderDetails(button.dataset.key);
    }
  });
});

renderDetails("q");
