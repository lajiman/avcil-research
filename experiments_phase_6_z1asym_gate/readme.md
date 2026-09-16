显式 Energy Gate：从问题设置到 Class-wise Logit Routing

1. 问题设置与符号

设类别增量学习一共有 (T) 个 incremental steps。

在 step (t)：

[\mathcal{Y}_t]

表示当前新增类别集合；

\bigcup_{\tau=0}^{t}\mathcal{Y}_\tau]

表示截至 step (t) 已经见过的全部类别。

对样本 (i)：

[x_i^a]

表示音频输入；

{x_{i,\tau,j}^v}]

表示视觉输入，其中：

(\tau=1,\ldots,T_v) 表示时间位置；

(j=1,\ldots,K) 表示空间位置；

(y_i\in\mathcal{C}_t) 表示类别标签。

模型包含三个主要表示层次：

[z_0:\text{预提取的固定音频、视觉特征},]

[z_1:\text{融合前的 audio 和 visual representation},]

[z_2:\text{audio 与 visual 相加后的融合表示}.]

本文关注的是：针对不同类别，audio-guided visual attention 是否都应该以相同强度作用于视觉分支。

2. 原始 Audio-guided Visual Attention

对于样本 (i)，模型利用音频特征计算视觉空间注意力和时间注意力。

空间注意力为：

\operatorname{Softmax}{j}\left(s{\mathrm{spa}}(x_i^a,x_{i,\tau,j}^v)\right),]

时间注意力为：

\operatorname{Softmax}{\tau}\left(s{\mathrm{temp}}(x_i^a,X_i^v)\right).]

于是，audio-guided visual pooling 为：

\sum_{\tau=1}^{T_v}\beta_{i,\tau}\sum_{j=1}^{K}\alpha_{i,\tau,j}x_{i,\tau,j}^v.]

经过 visual projection 后，得到 guided visual (z_1) 表示：

\operatorname{ReLU}\left(W_v\bar v_i^G+b_v\right).]

其中，上标 (G) 表示 guided。

音频 (z_1) 表示为：

\operatorname{ReLU}\left(W_ax_i^a+b_a\right).]

原始 AVCIL 的融合表示为：

a_i+v_i^G.]

随后通过线性分类器：

W_cz_{2,i}^{G}+b_c.]

对类别 (c)，对应 logit 为：

w_c^\top(a_i+v_i^G)+b_c.]

3. 构造不受音频指导的视觉分支

为了判断 audio guidance 是否会伤害 visual representation，需要构造一个不依赖 audio attention 的视觉分支，作为 counterfactual reference。

对视觉时空位置进行 uniform pooling：

\frac{1}{T_vK}\sum_{\tau=1}^{T_v}\sum_{j=1}^{K}x_{i,\tau,j}^v.]

再经过与 guided branch 相同的 visual projection：

\operatorname{ReLU}\left(W_v\bar v_i^U+b_v\right).]

其中，上标 (U) 表示 uniform 或 unguided。

因此，模型同时得到：

[v_i^G:\text{受到 audio-guided attention 影响的视觉表示},]

[v_i^U:\text{不依赖 audio guidance 的视觉表示}.]

当前方法不直接修改 raw spatial 或 temporal attention，而是在最终类别判别中控制 guided visual contribution 与 uniform visual contribution 的比例。

4. 在 (z_1) 空间定义模态难度

对模态：

[m\in{a,v},]

定义用于 difficulty estimation 的表示：

[z_{a,i}=a_i,]

[z_{v,i}=v_i^U.]

视觉难度使用 uniform visual branch，而不是 guided visual branch。原因是：如果使用 (v_i^G)，audio 已经改变了 visual representation，会形成如下循环：

[\text{audio 改变 visual representation}\rightarrow\text{用该 visual representation 估计 difficulty}\rightarrow\text{difficulty 再决定 audio gate}.]

首先对表示进行 (L_2) 归一化：

\frac{z_{m,i}}{|z_{m,i}|_2}.]

对于类别 (c)，类别中心定义为：

\operatorname{Norm}\left(\frac{1}{N_c}\sum_{i=c}\bar z_{m,i}\right),]

其中：

\left|{i=c}\right|.]

4.1 类内离散程度

使用 cosine distance 定义类别 (c) 在模态 (m) 中的类内离散程度：

\frac{1}{N_c}\sum_{i=c}\left[1-\bar z_{m,i}^{\top}\mu_{m,c}\right].]

其中：

(S_{m,c}) 越小，类别内部越紧凑；

(S_{m,c}) 越大，类别内部越分散。

4.2 最近类别中心距离

类别 (c) 与其他类别中最近类别中心的距离为：

\min_{c'\in\mathcal{C}t,;c'\neq c}\left[1-\mu{m,c}^{\top}\mu_{m,c'}\right].]

其中：

(B_{m,c}) 越大，类别与最近竞争类别之间分离越明显；

(B_{m,c}) 越小，类别越容易与其他类别混淆。

4.3 Normalized Margin

定义 normalized margin：

\frac{B_{m,c}}{S_{m,c}+\epsilon}}]

其中 (\epsilon>0) 用于数值稳定。

其含义为：

(M_{m,c}) 大：类内紧凑、类间分离，类别在该模态中较可靠；

(M_{m,c}) 小：类内分散或类间接近，类别在该模态中较困难。

也可以定义对应 difficulty：

\frac{S_{m,c}}{B_{m,c}+\epsilon}.]

在数值稳定项较小时：

[M_{m,c}\approx\frac{1}{D_{m,c}}.]

5. 使用 kNN Purity 判断困难是否具有结构

仅使用 normalized margin，无法区分以下两种情况：

类别整体较困难，但局部邻域结构仍然稳定；

表示空间本身已经接近无结构噪声。

因此进一步引入 kNN purity。

对于样本 (i)，在模态 (m) 的归一化表示空间中寻找 (k) 个最近邻：

[\mathcal N_k^m(i).]

类别 (c) 的 kNN purity 定义为：

\frac{\displaystyle\sum_{i=c}\sum_{j\in\mathcal N_k^m(i)}\mathbf 1[y_j=c]+\alpha}{N_ck+2\alpha}.]

其中：

(\mathbf 1[\cdot]) 是指示函数；

(\alpha>0) 是 Beta smoothing 参数；

当前默认可以取 (\alpha=1)。

加入 smoothing 的目的，是避免小样本类别由于偶然没有同类邻居而直接得到：

[P_{m,c}=0.]

kNN purity 的含义为：

(P_{m,c}) 高：类别局部邻域仍具有清晰结构；

(P_{m,c}) 低：局部样本大量被其他类别包围，更接近无结构噪声。

6. 定义模态结构可靠性

将 normalized margin 与 kNN purity 相结合，定义类别 (c) 在模态 (m) 中的结构可靠性：

M_{m,c}P_{m,c}}]

其中：

[R_{a,c}]

表示 audio 对类别 (c) 的结构可靠性；

[R_{v,c}]

表示 uniform visual 对类别 (c) 的结构可靠性。

该定义同时考虑：

全局类间与类内几何结构；

局部邻域结构。

如果一个类别具有较大的 normalized margin，但局部 purity 很低，则其可靠性仍会受到抑制。这可以避免将偶然形成的全局几何结构误判为稳定语义结构。

7. 定义 Interference Energy

我们关心的并不是某个模态的绝对 difficulty，而是：

对类别 (c)，audio 是否相对于 visual 更不可靠。

因此定义类别 (c) 的 interference energy observation：

\left[\log\frac{R_{v,c}+\epsilon}{R_{a,c}+\epsilon}\right]_+}]

其中：

\max(x,0).]

情况一：Audio 不弱于 Visual

当：

[R_{a,c}\geq R_{v,c},]

有：

[\widehat E_c=0.]

这表示没有足够证据说明 audio guidance 应该被削弱。

情况二：Audio 弱于 Visual

当：

[R_{a,c}<R_{v,c},]

有：

\log\frac{R_{v,c}}{R_{a,c}}

0.]

audio 相对越不可靠，energy 越大。

使用对数比值的原因是：

\log R_v-\log R_a,]

因此它描述的是相对比例差异，而不是绝对差值。

例如：

\frac{0.4}{0.2}

2,]

两种情况下都表示 visual reliability 是 audio reliability 的两倍，因此得到相同的 energy。

8. 从 Energy 得到显式 Gate

类别 (c) 的 guidance gate 定义为：

\exp(-E_c)}]

其中：

[E_c\geq0,\qquadg_c\in(0,1].]

在不考虑 continual learning 更新时，可以直接令：

[E_c=\widehat E_c.]

因此：

\left[\log\frac{R_{v,c}+\epsilon}{R_{a,c}+\epsilon}\right]_+\right).]

它等价于：

\min\left(1,\frac{R_{a,c}+\epsilon}{R_{v,c}+\epsilon}\right)}]

因此：

当 audio 不弱于 visual：[g_c=1;]

当 audio reliability 是 visual reliability 的 (70%)：[g_c\approx0.7;]

当 audio reliability 只有 visual reliability 的 (20%)：[g_c\approx0.2.]

为了避免极端值，可以限制：

[E_c\in[0,E_{\max}],]

从而得到：

[g_c\geqe^{-E_{\max}}.]

需要注意的是，gate 控制的是：

[\text{audio 对 visual representation 的指导强度},]

而不是删除 audio modality 本身。

9. Class-wise Logit Routing

9.1 为什么不能直接使用真实类别 Gate

最直接的写法是：

g_{y_i}v_i^G+(1-g_{y_i})v_i^U.]

然后：

w_c^\top\left(a_i+v_i^{\mathrm{mix}}\right)+b_c.]

训练时真实标签 (y_i) 已知，因此这种写法可以实现。

但在测试阶段，真实标签未知，无法在预测之前选择：

[g_{y_i}.]

如果测试时使用真实标签选择 gate，会产生 label leakage。

9.2 Class-wise Routing 的定义

为避免依赖真实标签，对每一个候选类别 (c)，使用该类别自己的 gate 计算对应 score：

g_cv_i^G+(1-g_c)v_i^U.]

类别 (c) 的 logit 为：

w_c^\top\left(a_i+v_i^{(c)}\right)+b_c}]

展开后：

w_c^\top a_i+g_cw_c^\top v_i^G+(1-g_c)w_c^\top v_i^U+b_c}]

矩阵形式为：

W_ca_i+\mathbf g\odot W_cv_i^G+(1-\mathbf g)\odot W_cv_i^U+b_c}]

其中：

[g_1,\ldots,g_{|\mathcal C_t|}].]

对同一个样本 (i)：

[\ell_{i,1}\text{ 使用 }g_1,]

[\ell_{i,2}\text{ 使用 }g_2,]

[\cdots]

[\ell_{i,C}\text{ 使用 }g_C.]

最终预测为：

\arg\max_{c\in\mathcal C_t}\ell_{i,c}.]

9.3 为什么该形式在 AVCIL 中成立

当前 AVCIL 使用加法融合：

a_i+v_i,]

并使用线性分类器：

W_cz_{2,i}+b_c.]

因此原始类别 (c) 的 logit 可以精确拆分为：

w_c^\top a_i+w_c^\top v_i^G+b_c.]

当前方法只是将 guided visual contribution：

[w_c^\top v_i^G]

替换为：

[g_cw_c^\top v_i^G+(1-g_c)w_c^\top v_i^U.]

所以它是对原始线性类别 score 的显式扩展。

当所有类别 gate 均为 1 时：

[g_c=1,\qquad\forall c,]

有：

w_c^\top(a_i+v_i^G)+b_c,]

即严格退化为原始 AVCIL。

9.4 对 Class-wise Routing 的准确解释

Class-wise logit routing 并不表示一个样本具有多个真实 gate。

更准确的解释是：

对同一个输入样本，每个候选类别的判别函数拥有自己的 guidance policy。

将公式改写为：

w_c^\top(a_i+v_i^U)+g_cw_c^\top(v_i^G-v_i^U)+b_c.]

其中：

[v_i^G-v_i^U]

表示 audio guidance 相对于 uniform visual branch 带来的 visual change。

因此：

[g_cw_c^\top(v_i^G-v_i^U)]

表示：

在判断输入是否属于类别 (c) 时，应该在多大程度上相信 audio guidance 带来的 visual-logit change。

9.5 该设计包含的额外假设

Energy (E_c) 是根据类别 (c) 的真实样本统计得到的，因此它直接支持的是：

对真正属于类别 (c) 的样本，audio 相对于 visual 的可靠性如何。

但 class-wise logit routing 会将 (g_c) 用于所有样本的类别 (c) logit，包括：

[y_i\neq c]

的负样本。

因此，它额外假设：

[\boxed{g_c\text{ 不只是类别 }c\text{ 正样本的属性，}}]

而是：

[\boxed{g_c\text{ 是类别 }c\text{ 整个判别函数的属性。}}]

该假设在数学上是自洽的，但并不是由 class-wise reliability definition 自动保证的。因此，在实验中应进一步比较：

[\text{oracle one-gate-per-sample}]

与：

[\text{class-wise logit routing},]

以判断 energy 本身是否有效，以及 class-wise routing 是否引入了额外误差。