# 第 2 章：表示 (Representation)

为了在计算层面对不确定性进行严谨建模，我们必须首先建立一套形式化的数学表示体系。本章系统阐述如何形式化表示不确定性。

我们从**信念程度（degree of belief）**的基本概念出发，证明一组符合直觉的理性公理如何推导出使用**概率分布（probability distributions）**来定量刻画不确定性。随后，我们讨论离散与连续变量的常见分布形态以及混合模型。由于真实世界决策问题通常涉及大量相互交织的变量，我们将重点介绍如何利用变量之间的**条件独立性（conditional independence）**，通过**贝叶斯网络（Bayesian Networks）**高效因式分解并紧凑表示高维联合概率分布。

---

## 2.1 信念程度与概率公理 (Degrees of Belief and Probability)

在涉及不确定性的问题中，核心需求是能够对不同陈述（命题）的**可信度（plausibility）**进行严格比较。例如，我们希望能够表达：命题 $A$ 比命题 $B$ 更为可信。若 $A$ 表示“执行机构失效”，而 $B$ 表示“传感器失效”，我们可以形式化记作：
$$
A \succ B
$$

基于这个基础偏序关系 $\succ$，我们可以定义其他衍生关系：
- $A \prec B \iff B \succ A$ （命题 $A$ 的可信度低于 $B$）
- $A \sim B \iff (A \not\succ B) \land (B \not\succ A)$ （命题 $A$ 与 $B$ 等度可信）
- $A \succeq B \iff (A \succ B) \lor (A \sim B)$ （命题 $A$ 不劣于 $B$）
- $A \preceq B \iff (A \prec B) \lor (A \sim B)$ （命题 $A$ 不优于 $B$）

### 理性比较的公理假设
为了保证信念比较的一致性，理性的决策主体必须满足以下公理：
1. **全序性（Completeness）**：对于任意两个命题 $A$ 与 $B$，必有且仅有 $A \succ B$、$A \prec B$ 或 $A \sim B$ 之一成立；
2. **传递性（Transitivity）**：若 $A \succ B$ 且 $B \succ C$，则必有 $A \succ C$。

传递性与全序性假设保证了我们可以用一个实数值来量化可信度。考克斯（Richard Cox, 1946）进一步证明，只要可信度测度满足基本的常识一致性原则，其数学运算规则在同构意义下**唯一等价于标准概率论**：
$$
P(A \mid C) \in [0, 1]
$$
且满足经典的柯尔莫哥洛夫公理：
1. 非负性：$P(A) \ge 0$；
2. 规范性：若 $S$ 为全集样本空间，则 $P(S) = 1$；
3. 可加性：若 $A$ 与 $B$ 互斥，则 $P(A \lor B) = P(A) + P(B)$。

::: tip 边注 2.1：荷兰赌论证 (Dutch Book Argument)
如果不确定性信念的赋值违反了概率公理，那么存在一种赌局组合（被称为荷兰赌，Dutch Book），使得无论现实发生何种结果，参与者都注定遭受确定的净亏损。因此，遵循概率公理是理性主体的内在自洽要求。
:::

---

## 2.2 概率分布 (Probability Distributions)

随机变量用大写字母表示（如 $X$），其具体取值用小写字母表示（如 $x$）。变量取值所在的空间称为**定义域（domain）**或样本空间。

### 2.2.1 离散随机变量
对于取值于有限或可数离散集合的随机变量 $X$，其不确定性通过**概率质量函数（Probability Mass Function, PMF）**刻画：
$$
P(X = x) = P(x) \ge 0, \quad \sum_{x} P(x) = 1
$$

![概率质量函数 PMF](/figures/fig_2_1.png)
*图 2.1：离散变量的概率质量函数 PMF 示意。每个可能取值对应离散概率高度，总和严格等于 1。*

### 2.2.2 连续随机变量与分位数
对于定义在连续空间（如实数集 $\mathbb{R}$）上的随机变量，单点的概率为零。我们通过**概率密度函数（Probability Density Function, PDF）** $p(x)$ 来描述概率：
$$
p(x) \ge 0, \quad \int_{-\infty}^{\infty} p(x)\, dx = 1
$$
任意区间 $[a, b]$ 内的概率由定积分给出：
$$
P(a \le X \le b) = \int_{a}^{b} p(x)\, dx
$$

![概率密度函数与累积分布函数](/figures/fig_2_2.png)
*图 2.2：连续随机变量的概率密度函数 PDF。*

**累积分布函数（Cumulative Distribution Function, CDF）**定义为事件 $X \le x$ 的概率：
$$
F(x) = P(X \le x) = \int_{-\infty}^{x} p(t)\, dt
$$
对于连续变量，$F(x)$ 单调不减且值域在 $[0, 1]$ 之间。其反函数被称为**分位数函数（Quantile Function）**：
$$
Q(\alpha) = F^{-1}(\alpha)
$$
分位数函数满足：变量 $X$ 取值小于等于 $Q(\alpha)$ 的概率恰好为 $\alpha$。

| ![CDF 曲线](/figures/fig_2_3.png) | ![分位数函数](/figures/fig_2_4.png) |
| :---: | :---: |
| *图 2.3：累积分布函数 CDF* | *图 2.4：标准高斯分布的分位数函数* |

### 2.2.3 常用连续分布与截断分布
- **高斯分布（正态分布）**：由均值 $\mu$ 和方差 $\sigma^2$ 确定：
  $$
  \mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
  $$
- **截断分布（Truncated Distributions）**：当物理状态受限于特定边界 $[a, b]$ 时（例如速度不能为负），可对分布在区间外截断并重新归一化。

![截断高斯与均匀分布](/figures/fig_2_5.png)
*图 2.5：单位高斯分布、截断高斯分布与均匀分布对比。*

### 2.2.4 混合模型 (Mixture Models)
现实中的复杂多模态分布可以通过加权组合若干个单模态基础分布得到。设有 $m$ 个成分分布 $p_i(x)$，权重为 $w_i \ge 0$ 且 $\sum_{i=1}^m w_i = 1$，则混合密度函数为：
$$
p(x) = \sum_{i=1}^m w_i p_i(x)
$$

---

## 2.3 联合概率分布 (Joint Distributions)

实际决策往往需要同时对多个相互关联的变量进行综合建模，记为多元随机变量集合 $\mathbf{X} = \{X_1, \dots, X_n\}$。

### 2.3.1 离散联合分布：因子（Factor）与赋值（Assignment）
在离散情况下，联合分布指定了每个完整的可能赋值元组 $(x_1, \dots, x_n)$ 的概率。若每个变量有 $r$ 个离散取值，则全联合概率表需要 $r^n - 1$ 个独立参数，面临严重的指数级**维度灾难（curse of dimensionality）**。

在算法实现中，我们定义 `Variable`、`Assignment`（变量赋值字典）以及 `Factor`（因子表）核心数据结构：

```julia
# 核心数据结构：变量、赋值与因子表 (来自官方 Julia 算法实现)
struct Variable
	name::Symbol
	r::Int # number of possible values
end

const Assignment = Dict{Symbol,Int}
const FactorTable = Dict{Assignment,Float64}

struct Factor
	vars::Vector{Variable}
	table::FactorTable
end

variablenames(ϕ::Factor) = [var.name for var in ϕ.vars]

select(a::Assignment, varnames::Vector{Symbol}) =
	Assignment(n=>a[n] for n in varnames)

function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec([Assignment(n=>v for (n,v) in zip(names, values))
    			for values in product((1:v.r for v in vars)...)])
end

function normalize!(ϕ::Factor)
	z = sum(p for (a,p) in ϕ.table)
	for (a,p) in ϕ.table
		ϕ.table[a] = p/z
	end
	return ϕ
end
####################
```

### 2.3.2 连续联合分布与多元高斯
对于 $n$ 维连续随机向量 $\mathbf{x} \in \mathbb{R}^n$，多元高斯分布由均值向量 $\boldsymbol{\mu} \in \mathbb{R}^n$ 与协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$ 决定：
$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$
协方差矩阵必须是对称正定的。其参数数量为 $n + n(n+1)/2 = O(n^2)$。

| ![多元高斯混合模型等高线](/figures/fig_2_6.png) | ![决策树参数化](/figures/fig_2_7.png) |
| :---: | :---: |
| *图 2.6：二维高斯混合模型的概率密度等高线* | *图 2.7：分段决策树条件密度表示* |

多元高斯混合模型（GMM）通过多个具有独立权重、均值和协方差的高斯分量组合，能够拟合任意复杂的连续几何分布（图 2.8）：

![3成分多元高斯混合模型采样分布](/figures/fig_2_8.png)
*图 2.8：包含三个成分的多元高斯混合模型采样散点与密度分布。*

---

## 2.4 条件概率分布 (Conditional Distributions)

决策与推理的本质是在观察到部分证据（Evidence）之后，更新对其余未知变量的概率评估。

### 2.4.1 条件概率与贝叶斯法则
根据条件概率的乘法定义：
$$
P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)} = \frac{P(x, y)}{\sum_{x'} P(x', y)}
$$
在此基础上推导得到现代人工智能推理的核心法则——**贝叶斯定理（Bayes' Rule）**：
$$
P(X = x \mid Y = y) = \frac{P(Y = y \mid X = x) P(X = x)}{P(Y = y)}
$$
式中：
- $P(X = x)$ 为**先验概率（Prior）**；
- $P(Y = y \mid X = x)$ 为**似然项（Likelihood）**；
- $P(Y = y)$ 为**边缘证据概率（Evidence / Marginal）**；
- $P(X = x \mid Y = y)$ 为观察到证据后的**后验概率（Posterior）**。

### 2.4.2 连续条件模型与线性高斯
当变量连续时，条件分布 $p(x \mid y)$ 可以参数化为依赖于输入 $y$ 的函数。例如**线性高斯模型（Linear Gaussian Model）**：
$$
p(x \mid y) = \mathcal{N}(x \mid m y + b, \sigma^2)
$$
其均值与输入 $y$ 呈线性关系，方差恒定（图 2.9）：

![线性高斯条件分布](/figures/fig_2_9.png)
*图 2.9：线性高斯模型 $p(x \mid y) = \mathcal{N}(x \mid 2y + 1, 1)$。*

### 2.4.3 离散响应连续输入的 Logit / Softmax 模型
当输出 $X$ 为离散状态（例如二分类 $X \in \{x_1, x_2\}$）而条件输入 $y$ 为连续实数时，通常采用 **Logit（Logistic Sigmoid）** 模型：
$$
P(x_1 \mid y) = \frac{1}{1 + \exp(-(\theta_1 + \theta_2 y))}
$$
参数 $\theta_2$ 决定了随 $y$ 变化的转换陡峭程度（图 2.10）：

![Logit 模型不同斜率变化曲线](/figures/fig_2_10.png)
*图 2.10：Logit 模型在不同斜率参数 $\theta_2$ 下的条件概率响应曲面。*

---

## 2.5 贝叶斯网络 (Bayesian Networks)

为了化解全联合分布参数指数爆炸的困境，朱迪亚·珀尔（Judea Pearl）提出了**贝叶斯网络（Bayesian Networks）**。

### 2.5.1 结构与因式分解定理
贝叶斯网络是一个**有向无环图（Directed Acyclic Graph, DAG）**，记为 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- 节点 $X_i \in \mathcal{V}$ 对应一个随机变量；
- 有向边 $(X_j, X_i) \in \mathcal{E}$ 表示从父节点 $X_j$ 到子节点 $X_i$ 的直接概率依赖关系；
- 每个节点 $X_i$ 关联一个局部的条件概率分布 $P(X_i \mid \text{Parents}(X_i))$。

**贝叶斯网络局部因式分解定理**指出，全联合概率分布可严格表示为所有局部条件概率的简单乘积：
$$
P(X_1, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
$$

### 2.5.2 参数紧凑度优势
若有 $n$ 个二值变量：
- 全联合概率分布需要 $2^n - 1$ 个独立参数；
- 若每个节点在贝叶斯网络中最多只有 $k$ 个父节点，则全网络参数量仅为 $n \cdot 2^k$。当 $k \ll n$ 时，模型从**指数复杂度骤降为线性复杂度**！

![卫星遥测监测贝叶斯网络](/figures/bn_satellite.png)
*图 2.11：卫星遥测故障监控贝叶斯网络示意图。节点包括轨道环境、电池故障、太阳翼指向与遥测传感器读数。*

---

## 2.6 条件独立性与 d-分离 (Conditional Independence & d-Separation)

### 2.6.1 条件独立性定义
若在给定变量集合 $\mathbf{C}$ 的条件下，变量 $\mathbf{A}$ 与 $\mathbf{B}$ 满足：
$$
P(\mathbf{A}, \mathbf{B} \mid \mathbf{C}) = P(\mathbf{A} \mid \mathbf{C}) P(\mathbf{B} \mid \mathbf{C})
$$
则称 $\mathbf{A}$ 与 $\mathbf{B}$ 在给定 $\mathbf{C}$ 时**条件独立**，记作 $(\mathbf{A} \perp \mathbf{B} \mid \mathbf{C})$。

### 2.6.2 三大基本图连接模式
判断图上的条件独立性依赖于三类三节点局部路径结构：

![三大基本图连接模式](/figures/d_separation_motifs.png)
*图 2.12：因果链、分叉与倒置分叉（碰撞）。*

1. **因果链（Causal Chain）**：$X \to Y \to Z$
   - 当中间节点 $Y$ **未被观测**时，信息可在 $X$ 与 $Z$ 之间流动（不独立）；
   - 当中间节点 $Y$ **已被观测**时，路径被阻断，$(X \perp Z \mid Y)$。
2. **共因分叉（Common Cause / Fork）**：$X \leftarrow Y \to Z$
   - 当根节点 $Y$ **未被观测**时，$X$ 与 $Z$ 相互关联；
   - 当根节点 $Y$ **已被观测**时，共同原因被控制，路径阻断，$(X \perp Z \mid Y)$。
3. **碰撞/倒置分叉（Collider / Inverted Fork / V-Structure）**：$X \to Y \leftarrow Z$
   - 当碰撞节点 $Y$ 及其所有后代节点都**未被观测**时，路径天然阻断，$(X \perp Z)$（边际独立）；
   - **反直觉特性**：一旦碰撞节点 $Y$ 或其任意后代节点**被观测**，路径被激活打通！$X$ 与 $Z$ 变为相关（这被称为“合理解释效应” Explaining Away）。

### 2.6.3 d-分离（定向分离）准则
对于网络中任意无向路径，若该路径上存在至少一个节点满足以下条件之一，则该路径被集合 $\mathbf{C}$ **阻断（blocked）**：
1. 该节点位于链式 $X \to Y \to Z$ 或分叉 $X \leftarrow Y \to Z$ 结构中，且 $Y \in \mathbf{C}$；
2. 该节点为碰撞节点 $X \to Y \leftarrow Z$，且 $Y \notin \mathbf{C}$ 且 $Y$ 的所有后代节点均不在 $\mathbf{C}$ 中。

若连接变量集合 $\mathbf{A}$ 与 $\mathbf{B}$ 之间的**所有可能无向路径**均被集合 $\mathbf{C}$ 阻断，则称 $\mathbf{A}$ 与 $\mathbf{B}$ 被 $\mathbf{C}$ **d-分离（d-separated）**，由此保证条件独立性 $(\mathbf{A} \perp \mathbf{B} \mid \mathbf{C})$ 恒成立。

![d-分离查询示例](/figures/bn_query.png)
*图 2.13：卫星网络中多重路径阻断判定示例。*

---

## 2.7 本章小结 (Summary)

- **概率论的一致性**：基于信念程度的偏序关系公理必然导向柯尔莫哥洛夫概率公理体系；
- **分布表示**：离散随机变量由 PMF 描述，连续变量由 PDF 描述；混合模型与截断分布提供了表达任意几何复杂度的灵活性；
- **高维联合分布的挑战**：直接参数化面临维度灾难，指数级参数量使得统计学习与存储不可行；
- **贝叶斯网络的威力**：利用 DAG 图结构显式编码变量间的局部条件独立性，通过局部因式分解定理实现联合分布的极紧凑线性/多项式表示；
- **d-分离判定法则**：通过因果链、分叉与碰撞节点的三大阻断条件，提供了纯图拓扑级别线性时间判定任意条件独立性结论的数学工具。

---

## 2.8 经典习题与详细解答 (Exercises & Solutions)

### 习题 2.1 (Exercise 2.1)
**题目**：假设你有三个离散变量 $X_1, X_2, X_3$，取值个数分别为 $r_1=2, r_2=3, r_3=4$。若不作任何独立性假设，指定其完全联合概率分布至少需要多少个独立参数？若三者互相独立，需要多少个独立参数？

**详细解答**：
1. **完全联合分布**：三变量的所有可能取值组合数为 $r_1 \times r_2 \times r_3 = 2 \times 3 \times 4 = 24$。因为所有取值概率之和必须为 1，所以独立参数个数为：
   $$
   24 - 1 = 23
   $$
2. **互相独立假设**：联合分布因式分解为 $P(X_1, X_2, X_3) = P(X_1)P(X_2)P(X_3)$。
   - $P(X_1)$ 需要 $r_1 - 1 = 2 - 1 = 1$ 个参数；
   - $P(X_2)$ 需要 $r_2 - 1 = 3 - 1 = 2$ 个参数；
   - $P(X_3)$ 需要 $r_3 - 1 = 4 - 1 = 3$ 个参数；
   总共仅需 $1 + 2 + 3 = 6$ 个独立参数。独立性假设使参数规模大幅缩减。

---

### 习题 2.8 (Exercise 2.8)
**题目**：给定如下贝叶斯网络拓扑结构：

![习题 2.8 贝叶斯网络](/figures/exercise_bn.png)
*图 2.14：习题 2.8 的贝叶斯网络图结构。*

在给定 $C$ 的条件下，变量 $A$ 是否与 $E$ 达成 d-分离，即是否满足 $(A \perp E \mid C)$？请详述判定路径。

**详细解答**：
连接 $A$ 与 $E$ 的路径主要包含：
1. 路径 $A \to B \to D \to E$：
   - 检查中间节点 $B$ 与 $D$：两者均不在条件观测集合 $\{C\}$ 中；
   - 路径中的节点不是碰撞节点，因此在未被观测时，该路径**保持连通激活**；
2. 结论：由于存在未被集合 $\{C\}$ 阻断的有向路径，$A$ 与 $E$ **不满足 d-分离**，故 $(A \not\perp E \mid C)$。
