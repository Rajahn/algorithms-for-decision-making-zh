# 第 2 章：表示 (Representation)

在计算层面对不确定性进行严谨建模，必须首先建立一套形式化的数学表示体系。本章系统阐释如何在计算机中精确表示不确定性。

我们将从**信念程度（degree of belief）**的基本概念出发，证明一组符合直觉的理性比较公理如何直接推导出利用**概率分布（probability distributions）**来定量刻画不确定性。随后，我们系统讨论离散与连续随机变量的经典分布形态及其混合模型（mixture models）。由于真实世界的决策问题通常涉及大量相互交织的随机变量，我们将重点阐述如何利用变量之间的**条件独立性（conditional independence）**，通过**贝叶斯网络（Bayesian Networks）**实现高维联合概率分布的极其紧凑的因子分解与算法表示。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（因果积木）**：若将 $n$ 个二值变量的联合分布写成一张大表格，需要 $2^n$ 行数据（当 $n=30$ 时超过 10 亿行，内存崩溃）。贝叶斯网络本质上是**“利用稀疏因果关系给高维空间拆积木”**——只要某个变量只依赖于它的少数几个直接原因（父节点），联合分布就能被拆解成一系列极小的局部条件概率表。
- **关键突破：d-分离（d-separation）**：判断两个变量在给定某些已知证据时是否“统计独立”，无需做复杂的微积分积分，只需看网络中有向路径是否被阻断：
  - **因果链（$A \to B \to C$）**：若中间原因 $B$ 已知，因果被阻断；
  - **共因分叉（$A \leftarrow B \to C$）**：若共同起因 $B$ 已知，两结果之间的虚假相关性被阻断；
  - **倒置分叉/碰撞节点（$A \to B \leftarrow C$）**：平时两者互相独立；但**一旦其共同结果 $B$ 被观测到，原本无关的两因就会产生关联（合理解释效应，Explaining Away）**！
- **避坑指北**：贝叶斯网络中的有向箭头虽然通常顺应人类直觉中的因果流向，但在纯统计意义上，它严格代表的是**条件概率因式分解顺序**，不同拓扑可能有完全相同的统计独立性。
:::
---

## 2.1 信念程度与概率公理 (Degrees of Belief and Probability)

在涉及不确定性的问题中，核心需求是能够对不同陈述（命题）的**可信度（plausibility）**进行严格比较。例如，我们希望能够形式化表达：“命题 $A$ 比命题 $B$ 更为可信”。若命题 $A$ 表示“执行机构发生故障”，而命题 $B$ 表示“传感器发生故障”，我们将其数学记作：
$$
A \succ B
$$

基于这个基础的二元偏序关系 $\succ$，我们可以形式化衍生定义其他比较关系：
$$
\begin{aligned}
A \prec B &\iff B \succ A \quad &(\text{命题 } A \text{ 的可信度低于 } B) \\
A \sim B &\iff (A \not\succ B) \land (B \not\succ A) \quad &(\text{命题 } A \text{ 与 } B \text{ 等度可信}) \\
A \succeq B &\iff (A \succ B) \lor (A \sim B) \quad &(\text{命题 } A \text{ 不劣于 } B) \\
A \preceq B &\iff (A \prec B) \lor (A \sim B) \quad &(\text{命题 } A \text{ 不优于 } B)
\end{aligned}
$$

### 理性信念公理
为了保证主观信念评估的一致性与自洽性，理性的决策主体必须遵循以下两条基础公理：
1. **完备性（Completeness）**：对于任意两个命题 $A$ 与 $B$，必有且仅有 $A \succ B$、$A \prec B$ 或 $A \sim B$ 之一成立；
2. **传递性（Transitivity）**：若 $A \succ B$ 且 $B \succ C$，则必有 $A \succ C$。

完备性与传递性公理保证了我们可以用一个实数值来连续量化命题的可信度。理查德·考克斯（Richard Cox, 1946）进一步证明：只要可信度测度在代数运算上满足基本的常识一致性原则，其数学运算规则在同构意义下**唯一等价于标准概率论法则**：
$$
P(A \mid C) \in [0, 1]
$$
该实数值满足著名的柯尔莫哥洛夫（Kolmogorov）概率公理体系：
1. **非负性**：对任意事件 $A$，$P(A) \ge 0$；
2. **规范性**：若 $S$ 为样本空间全集，则 $P(S) = 1$；
3. **可加性**：若事件 $A$ 与 $B$ 互斥（$A \cap B = \emptyset$），则 $P(A \cup B) = P(A) + P(B)$。

::: info 边注 2.1：荷兰赌论证 (Dutch Book Argument)
如果不确定性信念的数值赋值违反了柯尔莫哥洛夫公理，那么存在一种由若干下注构成的组合赌局（即“荷兰赌”），使得无论现实发生何种结果，下注者都注定承受确定的净亏损。因此，概率公理并非人为强加的主观设定，而是理性决策主体避免确定性破产的唯一数学自洽选择。
:::

::: info 边注 2.2：频率学派与贝叶斯学派
在概率的哲学诠释上存在两大经典流派：
- **频率学派（Frequentist）**：将概率诠释为可无限重复随机试验中事件发生频率的极限；
- **贝叶斯学派（Bayesian）**：将概率诠释为主体在面对不完全信息时对特定命题真伪的主观信念程度（degree of belief）。在自主决策智能体中，贝叶斯诠释天然适用于无法无限次重复的单次决策场景。
:::

---

## 2.2 概率分布 (Probability Distributions)

随机变量用大写字母表示（如 $X$），其具体的实值取值用对应的小写字母表示（如 $x$）。变量所有可能取值的集合称为该变量的**定义域（domain）**或样本空间。

### 2.2.1 离散随机变量与 PMF
对于取值于有限或可数离散集合的随机变量 $X$，其概率规律通过**概率质量函数（Probability Mass Function, PMF）**进行刻画：
$$
P(X = x) = P(x) \ge 0, \quad \sum_{x} P(x) = 1
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_1.png" alt="离散随机变量的 PMF 柱状图" style="max-width: 280px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.1：六面不均匀骰子的概率质量函数（PMF）示意图。</p>
</div>

::: tip 例 2.1：离散概率分布（六面不均匀骰子）
设随机变量 $X \in \{1, 2, 3, 4, 5, 6\}$ 代表投掷一枚不均匀骰子的点数，其 PMF 如下表所示：

| 点数 $x$ | 1 | 2 | 3 | 4 | 5 | 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $P(x)$ | 0.05 | 0.10 | 0.15 | 0.15 | 0.25 | 0.30 |

易见所有取值非负且 $\sum_{x=1}^6 P(x) = 0.05 + 0.10 + 0.15 + 0.15 + 0.25 + 0.30 = 1.0$。
:::

### 2.2.2 连续随机变量、PDF、CDF 与分位数
对于定义在实数连续空间（如 $\mathbb{R}$）上的随机变量，单点的概率测度为零。我们通过**概率密度函数（Probability Density Function, PDF）** $p(x)$ 来描述其概率分布：
$$
p(x) \ge 0, \quad \int_{-\infty}^{\infty} p(x)\, dx = 1
$$
变量落在任意区间 $[a, b]$ 内的概率由定积分给出：
$$
P(a \le X \le b) = \int_{a}^{b} p(x)\, dx
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_2.png" alt="概率密度函数曲线与微分概率" style="max-width: 260px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.2：连续随机变量的概率密度函数（PDF）。阴影微元面积 $p(x)dx$ 表示微小区间内的概率。</p>
</div>

**累积分布函数（Cumulative Distribution Function, CDF）**定义为事件 $X \le x$ 的概率：
$$
F(x) = P(X \le x) = \int_{-\infty}^{x} p(t)\, dt
$$
对于连续变量，$F(x)$ 是单调不减的连续函数，且 $\lim_{x \to -\infty} F(x) = 0$，$\lim_{x \to \infty} F(x) = 1$。其反函数称为**分位数函数（Quantile Function）**：
$$
Q(\alpha) = F^{-1}(\alpha), \quad \alpha \in [0, 1]
$$
分位数函数满足：变量 $X$ 取值小于等于 $Q(\alpha)$ 的概率恰好为 $\alpha$。例如 $Q(0.5)$ 即为分布的**中位数（median）**。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_2_3.png" alt="累积分布函数 CDF" style="max-width: 270px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.3：连续随机变量的累积分布函数（CDF）。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_2_4.png" alt="标准高斯分位数函数" style="max-width: 270px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.4：标准正态分布的分位数函数 $Q(\alpha)$。</p>
  </div>
</div>

### 2.2.3 常用连续分布与截断分布
- **均匀分布（Uniform Distribution）**：在区间 $[a, b]$ 上具有恒定密度：
  $$
  \mathcal{U}(x \mid a, b) = \frac{1}{b - a} \quad (a \le x \le b)
  $$
- **高斯分布（正态分布，Gaussian Distribution）**：由均值 $\mu$ 与方差 $\sigma^2$ 完全确定：
  $$
  \mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
  $$
- **截断分布（Truncated Distributions）**：在物理世界中，许多变量具有天然的硬边界（例如机器人的速度、温度等不可为负或受机械限位）。若将分布约束在区间 $[a, b]$ 内，必须对区间外置零并在区间内重新归一化：
  $$
  p_{[a, b]}(x) = \frac{p(x)}{\int_{a}^{b} p(t)\, dt} = \frac{p(x)}{F(b) - F(a)} \quad (a \le x \le b)
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_5.png" alt="单位高斯分布、截断高斯与均匀分布对比" style="max-width: 280px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.5：标准正态分布（黑色虚线）、区间 $[-1, 1]$ 上的截断正态分布（蓝色实线）以及均匀分布（灰色实线）对比。</p>
</div>

### 2.2.4 混合模型 (Mixture Models)
现实问题中的很多物理分布表现出多模态（multimodal）特征。通过对 $m$ 个简单的基础成分密度 $p_i(x)$ 进行线性凸组合，可以构建表达能力极强的**混合模型（Mixture Model）**：
$$
p(x) = \sum_{i=1}^m w_i p_i(x), \quad w_i \ge 0, \quad \sum_{i=1}^m w_i = 1
$$
式中 $w_i$ 为第 $i$ 个成分的混合权重。

::: tip 例 2.2：双峰高斯混合模型
设某传感器测量误差服从两个高斯分布的加权混合：
$$
p(x) = 0.4 \cdot \mathcal{N}(x \mid -2, 0.5^2) + 0.6 \cdot \mathcal{N}(x \mid 2, 1.0^2)
$$
由于 $0.4 + 0.6 = 1$，该分布是一个有效概率密度，具有明显的两个概率波峰。
:::

---

## 2.3 联合概率分布 (Joint Distributions)

实际决策往往需要同时对多个相互关联的变量进行综合建模，记为多元随机变量集合 $\mathbf{X} = \{X_1, \dots, X_n\}$。

### 2.3.1 离散联合分布：因子（Factor）与赋值（Assignment）
在离散情况下，联合分布指定了每个完整的赋值元组 $(x_1, \dots, x_n)$ 的概率值。若变量 $X_i$ 具有 $r_i$ 个可能离散取值，则所有可能的赋值组合数为 $\prod_{i=1}^n r_i$。因所有取值概率之和必须为 1，故完全指定该联合分布需要：
$$
\left( \prod_{i=1}^n r_i \right) - 1
$$
个独立参数。若每个变量均有 $r$ 个状态，参数量为 $O(r^n)$，随着变量数量的增加面临极其严峻的**维度灾难（curse of dimensionality）**。

在官方 Julia 算法实现中，我们定义 `Variable`（包含符号名称与取值数 $r$）、`Assignment`（变量取值字典）以及 `Factor`（包含作用域变量列表与概率查找表的因子）作为核心数据结构：

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

# 辅助转换：支持利用 NamedTuple 快速构造变量赋值字典
Dict{K,V}(a::NamedTuple) where K where V =
    Dict{K,V}(n=>v for (n,v) in zip(keys(a), values(a)))
Base.convert(::Type{Dict{K,V}}, a::NamedTuple) where K where V = Dict{K,V}(a)
Base.isequal(a::Dict{<:Any,<:Any}, nt::NamedTuple) =
    length(a) == length(nt) && all(a[n] == v for (n,v) in zip(keys(nt), values(nt)))
####################
```

::: tip 例 2.3：二元离散因子表
设变量 $X$ 与 $Y$ 均为二值变量（取值为 1 或 2）。因子 $\phi(X, Y)$ 的查找表包含 $2 \times 2 = 4$ 个条目：
$$
\begin{aligned}
\phi(X=1, Y=1) &= 0.1 \\
\phi(X=1, Y=2) &= 0.2 \\
\phi(X=2, Y=1) &= 0.3 \\
\phi(X=2, Y=2) &= 0.4
\end{aligned}
$$
其各条目求和为 $0.1+0.2+0.3+0.4 = 1.0$，构成合法的归一化联合概率分布。
:::

### 2.3.2 连续多元分布与多元高斯
对于 $n$ 维连续随机向量 $\mathbf{x} = [x_1, \dots, x_n]^\top \in \mathbb{R}^n$，最常用的参数化模型是**多元高斯分布（Multivariate Gaussian）**，由均值向量 $\boldsymbol{\mu} \in \mathbb{R}^n$ 与协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$ 确定：
$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$
协方差矩阵 $\boldsymbol{\Sigma}$ 必须是对称正定矩阵（symmetric positive definite）。多元高斯分布的独立参数个数为均值向量的 $n$ 个加上协方差矩阵对角线与上半部分的 $n(n+1)/2$ 个，总计 $O(n^2)$ 个参数。

多元高斯同样可以扩展为**多元高斯混合模型（GMM）**：
$$
p(\mathbf{x}) = \sum_{i=1}^m w_i \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_2_6.png" alt="多元高斯混合模型等高线" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.6：二维高斯混合模型概率密度等高线图。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_2_7.png" alt="紧凑决策树表示" style="max-width: 220px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.7：分段常数密度的等价紧凑决策树表示。</p>
  </div>
</div>

下图 2.8 展示了一个由三个多元高斯分量混合而成的 GMM 模型及其各单分量采样分布：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_8.png" alt="3个成分高斯混合模型全景" style="max-width: 720px; width: 100%; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.8：三成分多元高斯混合模型全景。前三图分别为分量 1、分量 2、分量 3 的采样与等高线，第四图为加权合成后的最终混合密度分布。</p>
</div>

---

## 2.4 条件概率分布 (Conditional Distributions)

智能体在环境中持续行动的核心环节，是在获取新的观测证据后，更新对未知状态的信念评估。

### 2.4.1 条件概率与贝叶斯法则
根据条件概率的乘法定义，给定事件 $Y=y$ 时，事件 $X=x$ 的条件概率为：
$$
P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)} = \frac{P(x, y)}{\sum_{x'} P(x', y)}
$$
利用乘法规则 $P(x, y) = P(y \mid x) P(x) = P(x \mid y) P(y)$，可推导出著名的**贝叶斯定理（Bayes' Rule）**：
$$
P(X = x \mid Y = y) = \frac{P(Y = y \mid X = x) P(X = x)}{P(Y = y)} = \frac{P(Y = y \mid X = x) P(X = x)}{\sum_{x'} P(Y = y \mid X = x') P(X = x')}
$$
在上述公式中：
- $P(X = x)$ 称为**先验概率（Prior）**，代表观察到证据前对 $X$ 的信念；
- $P(Y = y \mid X = x)$ 称为**似然项（Likelihood）**，反映在假设状态下产生观测的概率；
- $P(Y = y)$ 称为**边际证据概率（Marginal Evidence）**，充当归一化常数；
- $P(X = x \mid Y = y)$ 称为**后验概率（Posterior）**，代表融合证据后更新的信念。

### 2.4.2 连续条件模型与线性高斯
对于连续随机变量，条件概率密度同样满足：
$$
p(x \mid y) = \frac{p(x, y)}{p(y)} = \frac{p(y \mid x) p(x)}{\int p(y \mid x') p(x')\, dx'}
$$
在连续控制中最常用的条件参数化模型是**线性高斯模型（Linear Gaussian Model）**，其条件均值与条件输入 $y$ 呈线性关系，方差保持恒定：
$$
p(x \mid y) = \mathcal{N}(x \mid m y + b, \sigma^2)
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_9.png" alt="线性高斯条件密度分布" style="max-width: 300px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.9：线性高斯模型 $p(x \mid y) = \mathcal{N}(x \mid 2y + 1, 1)$。沿任意给定 $y$ 轴剖面均为标准高斯分布。</p>
</div>

### 2.4.3 离散输出连续输入的 Logit / Softmax 模型
当子节点 $X$ 为二值离散状态（$X \in \{x_1, x_2\}$）而父节点 $y$ 为连续实数变量时，最常用的条件模型是 **Logit（Logistic Sigmoid）模型**：
$$
P(X = x_1 \mid y) = \frac{1}{1 + \exp(-(\theta_1 + \theta_2 y))}, \quad P(X = x_2 \mid y) = 1 - P(X = x_1 \mid y)
$$
参数 $\theta_2$ 控制了随着输入 $y$ 变化，概率发生跃迁的陡峭程度：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_2_10.png" alt="Logit 模型条件概率曲面" style="max-width: 280px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.10：Logit 模型在不同斜率参数 $\theta_2$ 下的条件响应曲线。</p>
</div>

若离散变量有 $r > 2$ 个可能取值，则自然推广为 **Softmax 模型**：
$$
P(X = x_i \mid y) = \frac{\exp(\theta_{i1} + \theta_{i2} y)}{\sum_{j=1}^r \exp(\theta_{j1} + \theta_{j2} y)}
$$

---

## 2.5 贝叶斯网络 (Bayesian Networks)

为了克服高维完全联合分布的维度灾难，朱迪亚·珀尔（Judea Pearl, 1988）引入了**贝叶斯网络（Bayesian Networks）**。

### 2.5.1 结构定义与因式分解定理
贝叶斯网络由两部分组成：
1. **网络拓扑结构 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$**：一个**有向无环图（Directed Acyclic Graph, DAG）**，其中每个顶点 $X_i \in \mathcal{V}$ 对应一个随机变量，有向边 $(X_j, X_i) \in \mathcal{E}$ 表示从父变量 $X_j$ 到子变量 $X_i$ 的直接概率因果依赖；
2. **局部条件概率分布集**：为网络中的每个变量节点 $X_i$ 指定其在父节点集合 $\text{Parents}(X_i)$ 条件下的局部概率分布 $P(X_i \mid \text{Parents}(X_i))$。若节点没有父节点，则为先验边际概率 $P(X_i)$。

**贝叶斯网络链式因式分解定理**表明：网络中所有变量的联合概率分布，恒等于所有节点的局部条件概率分布之连乘积：
$$
P(X_1, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))
$$

```julia
# 贝叶斯网络数据结构与联合概率链式求值算法
struct BayesianNetwork
	vars::Vector{Variable}
    factors::Vector{Factor}
	graph::SimpleDiGraph{Int64}
end
####################

function probability(bn::BayesianNetwork, assignment)
    subassignment(ϕ) = select(assignment, variablenames(ϕ))
    probability(ϕ) = get(ϕ.table, subassignment(ϕ), 0.0)
    return prod(probability(ϕ) for ϕ in bn.factors)
end
####################
```

### 2.5.2 案例分析：卫星遥测监控网络
考虑一个航天卫星轨道状态故障诊断问题，涉及以下五个二值随机变量：
- $B$：蓄电池故障（Battery failure）
- $S$：太阳翼帆板指向失效（Solar panel failure）
- $E$：整星主配电系统故障（Electrical system failure）
- $D$：下行数据遥测信号中断（Telemetry data stream down）
- $C$：通信收发机载波丢失（Communication carrier lost）

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/bn_satellite.png" alt="卫星遥测贝叶斯网络" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.11：卫星遥测故障诊断贝叶斯网络拓扑图。</p>
</div>

根据拓扑结构，五个变量的全局联合分布被高度因式分解为：
$$
P(B, S, E, D, C) = P(B) P(S) P(E \mid B, S) P(D \mid E) P(C \mid E)
$$

### 参数存储量对比：
- **无条件假设的全联合分布**：5 个二值变量共有 $2^5 = 32$ 种状态组合，需要 $32 - 1 = \mathbf{31}$ 个独立参数；
- **利用贝叶斯网络分解**：
  - $P(B)$：1 个参数
  - $P(S)$：1 个参数
  - $P(E \mid B, S)$：$2 \times 2 = 4$ 种父节点组合，每组 1 个独立参数，共 4 个参数
  - $P(D \mid E)$：2 种父节点状态，共 2 个参数
  - $P(C \mid E)$：2 种父节点状态，共 2 个参数
  总共仅需 $1 + 1 + 4 + 2 + 2 = \mathbf{10}$ 个独立参数！**参数量缩减了近 70%**。当变量达到上百个时，贝叶斯网络可以将原本天文数字般的参数规模压缩至多项式甚至线性复杂度。

---

## 2.6 条件独立性与 d-分离 (Conditional Independence & d-Separation)

### 2.6.1 条件独立性数学定义
若在已知观测变量集合 $\mathbf{C}$ 的条件下，变量集合 $\mathbf{A}$ 与 $\mathbf{B}$ 满足：
$$
P(\mathbf{A}, \mathbf{B} \mid \mathbf{C}) = P(\mathbf{A} \mid \mathbf{C}) P(\mathbf{B} \mid \mathbf{C})
$$
则称 $\mathbf{A}$ 与 $\mathbf{B}$ 在给定 $\mathbf{C}$ 时**条件独立（Conditionally Independent）**，形式化记作：
$$
(\mathbf{A} \perp \mathbf{B} \mid \mathbf{C})
$$

### 2.6.2 三类基础三节点连接基元
判断图拓扑中是否存在条件独立性，取决于路径上任意三节点局部的连接形态：

1. **因果链（Causal Chain）**：$X \to Y \to Z$
   - 联合分布分解为 $P(X, Y, Z) = P(X)P(Y \mid X)P(Z \mid Y)$；
   - 若中间节点 $Y$ **未被观测**，路径畅通，信息可自由流动，$X$ 与 $Z$ 具有统计相关性；
   - 若中间节点 $Y$ **已被观测**（$Y \in \mathbf{C}$），路径被**阻断（blocked）**，$(X \perp Z \mid Y)$ 成立。
2. **共因分叉（Common Cause / Fork）**：$X \leftarrow Y \to Z$
   - 联合分布分解为 $P(X, Y, Z) = P(Y)P(X \mid Y)P(Z \mid Y)$；
   - 若共同原因 $Y$ **未被观测**，$X$ 与 $Z$ 呈现表面相关；
   - 若共同原因 $Y$ **已被观测**（$Y \in \mathbf{C}$），路径被阻断，$(X \perp Z \mid Y)$ 成立。
3. **倒置分叉/碰撞节点（Collider / Inverted Fork / V-Structure）**：$X \to Y \leftarrow Z$
   - 联合分布分解为 $P(X, Y, Z) = P(X)P(Z)P(Y \mid X, Z)$；
   - **反直觉特性**：若碰撞节点 $Y$ 及其所有后代节点都**未被观测**，路径天生处于**阻断状态**，此时 $X$ 与 $Z$ 边际独立（$(X \perp Z)$）；
   - **然而，一旦碰撞节点 $Y$ 或其任一后代节点被观测**，路径瞬间被**激活连通**！$X$ 与 $Z$ 变为条件相关！

::: tip 例 2.7：合理解释效应 (Explaining Away)
在卫星网络中，考虑从蓄电池故障 $B$ 经由配电故障 $E$ 到太阳翼故障 $S$ 的倒置分叉路径：$B \to E \leftarrow S$。
- 在没有任何观测时，电池是否老化与太阳翼是否被陨石击中是两个完全独立的物理事件，$(B \perp S)$；
- 假若地面遥测确认卫星配电系统发生断电故障（观测到 $E$）；
- 此时如果我们进一步排查确认蓄电池电压正常（即没有发生电池故障 $\neg B$），那么根据因果关系，我们必然极大概率断定发生了太阳翼故障 $S$（因为太阳翼故障成为了断电的唯一解释）；
- 反之，如果我们确认发生了电池故障 $B$，则太阳翼故障的怀疑程度大幅下降。观测到其中一个原因，在逻辑上“解释掉（explains away）”了另一可能原因，导致 $B$ 与 $S$ 在给定 $E$ 的条件下强烈相关！
:::

### 2.6.3 d-分离（定向分离）准则
对于贝叶斯网络中任意两个变量节点之间的任意无向路径 $p$，若路径上存在某个节点 $V$ 满足以下两个条件之一，则称路径 $p$ 被观测集合 $\mathbf{C}$ **阻断（blocked）**：
1. 节点 $V$ 位于链式结构 $X \to V \to Z$ 或分叉结构 $X \leftarrow V \to Z$ 中，且 $V \in \mathbf{C}$；
2. 节点 $V$ 是碰撞节点 $X \to V \leftarrow Z$，且 $V \notin \mathbf{C}$，并且 $V$ 的所有后代节点也均不在 $\mathbf{C}$ 中。

**d-分离判定定理**：若连接变量集合 $\mathbf{A}$ 与 $\mathbf{B}$ 之间的**所有可能无向路径**全部被集合 $\mathbf{C}$ 阻断，则称 $\mathbf{A}$ 与 $\mathbf{B}$ 在给定 $\mathbf{C}$ 下被 **d-分离（d-separated）**，由此保证条件独立性 $(\mathbf{A} \perp \mathbf{B} \mid \mathbf{C})$ 在该网络拓扑诱导的所有参数化联合分布中恒成立。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/bn_query.png" alt="d-分离查询示例图" style="max-width: 300px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.12：示例 2.8 中包含 6 个节点的条件独立性判定贝叶斯网络。</p>
</div>

::: tip 例 2.8：d-分离判定全流程
判断上图 2.12 所示网络结构是否保证 $(D \perp B \mid F)$ 条件独立？
从节点 $D$ 到节点 $B$ 存在两条无向路径：
1. **第一条路径 $D \leftarrow A \to C \leftarrow B$**：
   - 包含分叉结构 $D \leftarrow A \to C$ 和倒置分叉结构 $A \to C \leftarrow B$；
   - 节点 $A$ 没有被观测（$A \notin \{F\}$），故分叉未被阻断；
   - 碰撞节点 $C$ 本身未在条件集中，但其后代节点 $F \in \{F\}$ 被观测，导致倒置分叉路径被激活！
   - 故第一条路径**处于连通状态**，未被阻断。
2. **第二条路径 $D \to E \leftarrow C \leftarrow B$**：
   - 碰撞节点 $E$ 的后代节点 $F$ 被观测，倒置分叉 $D \to E \leftarrow C$ 被激活；
   - 链式节点 $C$ 未被观测，未能阻断信息流动；
   - 故第二条路径同样**处于连通状态**。

**判定结论**：由于存在未被阻断的连通路径，网络拓扑结构**并不蕴含** $(D \perp B \mid F)$。
:::

---

## 2.7 本章小结 (Summary)

- **概率论的一致性**：基于信念程度的完备性与传递性公理，必然在数学同构意义下推导出柯尔莫哥洛夫概率公理体系；
- **概率分布的表达**：离散随机变量通过 PMF 描述，连续随机变量通过 PDF 与 CDF 描述；通过分位数函数 $Q(\alpha) = F^{-1}(\alpha)$ 可以进行逆概率推断；截断分布与混合模型提供了拟合任意几何复杂多模态密度的强大表达能力；
- **高维联合分布与维度灾难**：直接参数化离散联合分布具有 $O(r^n)$ 的指数级空间复杂度，多元高斯具有 $O(n^2)$ 复杂度；
- **贝叶斯网络的威力**：利用 DAG 图结构显式编码变量间的局部条件独立性，通过局部因式分解定理 $P(X_{1:n}) = \prod_i P(X_i \mid \text{Parents}(X_i))$ 将高维分布拆解为紧凑局部因子的乘积；
- **d-分离判定法则**：通过因果链、共因分叉与碰撞节点的三大阻断条件，提供了在图拓扑级别多项式时间内精确判定任意条件独立性结论的数学工具。

---

## 2.8 经典习题与官方详细解答 (Exercises & Solutions)

### 习题 2.1 (Exercise 2.1)
**题目**：考虑一个服从参数为 $\lambda$ 的指数分布连续随机变量 $X$，其在非负实数区间上的概率密度函数为 $p(x \mid \lambda) = \lambda \exp(-\lambda x)$（$x \ge 0$）。求 $X$ 的累积分布函数（CDF）。

**详细解答**：
根据 CDF 的定义，对概率密度函数从下限 $0$ 进行积分：
$$
F(x) = \int_{0}^{x} p(t \mid \lambda)\, dt = \int_{0}^{x} \lambda e^{-\lambda t}\, dt = \left[ -e^{-\lambda t} \right]_{0}^{x} = -e^{-\lambda x} - (-e^{0}) = 1 - e^{-\lambda x}
$$
故当 $x \ge 0$ 时，$F(x) = 1 - \exp(-\lambda x)$；当 $x < 0$ 时，$F(x) = 0$。

---

### 习题 2.2 (Exercise 2.2)
**题目**：对于图 2.6 所示的二维混合密度函数，写出其 5 个均匀分布混合成分的定义域区间。（存在多种有效解）。

**详细解答**：
一种标准的五成分矩形均匀分布划分方案为：
1. $\mathcal{U}([-10, -10], [-5, 10])$
2. $\mathcal{U}([-5, 0], [0, 10])$
3. $\mathcal{U}([-5, -10], [0, 0])$
4. $\mathcal{U}([0, -10], [10, 5])$
5. $\mathcal{U}([0, 5], [10, 10])$

---

### 习题 2.3 (Exercise 2.3)
**题目**：给定如下 $P(X, Y, Z)$ 的完整联合概率表，构造其等价的紧凑决策树表示：

| $X$ | $Y$ | $Z$ | $P(X, Y, Z)$ |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0.13 |
| 0 | 0 | 1 | 0.02 |
| 0 | 1 | 0 | 0.05 |
| 0 | 1 | 1 | 0.02 |
| 1 | 0 | 0 | 0.13 |
| 1 | 0 | 1 | 0.01 |
| 1 | 1 | 0 | 0.05 |
| 1 | 1 | 1 | 0.17 |
| 2 | 0 | 0 | 0.13 |
| 2 | 0 | 1 | 0.03 |
| 2 | 1 | 0 | 0.05 |
| 2 | 1 | 1 | 0.21 |

**详细解答**：
观察表中具有相同概率值的条目：
- 当 $Y=0, Z=0$ 时，无论 $X$ 取 0、1 还是 2，概率恒为 $0.13$；
- 当 $Y=1, Z=0$ 时，无论 $X$ 取 0、1 还是 2，概率恒为 $0.05$；

因此，在根节点优先判定 $Z=0$：
- 若 $Z=0$：进一步检查 $Y$：若 $Y=0$，叶节点输出 $0.13$；若 $Y=1$，叶节点输出 $0.05$（无需测试变量 $X$）；
- 若 $Z=1$：树分支分别对 $(X, Y)$ 进行展开测试，输出对应的其余概率值。这一紧凑决策树有效消除了 6 个重复叶节点。

---

### 习题 2.4 (Exercise 2.4)
**题目**：假设我们需要指定一个包含 3 个分量的 4 维多元高斯混合模型。我们要求其中 2 个高斯分量假设 4 个变量之间互相独立，而剩余 1 个高斯分量允许变量间存在任意相关性。请问该混合模型总共需要多少个独立参数？

**详细解答**：
1. **成分权重**：3 个混合权重满足归一化条件，需要 $3 - 1 = 2$ 个独立参数；
2. **前两个独立高斯分量**：
   - 变量互相独立意味着协方差矩阵为对角矩阵，仅需指定对角线上的 4 个方差；
   - 均值向量包含 4 个均值；
   - 每个独立分量需要 $4 + 4 = 8$ 个参数，两分量共 $2 \times 8 = 16$ 个参数；
3. **第三个完全相关高斯分量**：
   - 均值向量需要 4 个参数；
   - 任意对称正定协方差矩阵需要 $4 \times (4+1) / 2 = 10$ 个参数；
   - 该分量共需 $4 + 10 = 14$ 个参数；
4. **总参数量**：$2 + 16 + 14 = \mathbf{32}$ 个独立参数。

---

### 习题 2.5 (Exercise 2.5)
**题目**：设有三个相互独立的变量 $X_1, X_2, X_3$，分别由具有 4、7 和 3 个分箱边界（bin edges）的分段常数密度函数定义。指定它们的联合概率分布总共需要多少个独立参数？

**详细解答**：
具有 $m$ 个分箱边界的分段常数密度函数包含 $m - 1$ 个分箱区间。由于每个一维密度的总积分必须为 1，故各自分布需要 $(m - 1) - 1 = m - 2$ 个独立参数：
- $X_1$ 需要 $4 - 2 = 2$ 个独立参数；
- $X_2$ 需要 $7 - 2 = 5$ 个独立参数；
- $X_3$ 需要 $3 - 2 = 1$ 个独立参数；

由于三者互相独立，总独立参数个数为 $2 + 5 + 1 = \mathbf{8}$ 个。

---

### 习题 2.6 (Exercise 2.6)
**题目**：设有四个连续随机变量 $X_1, X_2, Y_1, Y_2$，我们需要构建线性高斯条件模型 $p(\mathbf{X} \mid \mathbf{Y})$，其中 $\mathbf{X} = [X_1, X_2]^\top$，$\mathbf{Y} = [Y_1, Y_2]^\top$。该模型形式为 $p(\mathbf{X} \mid \mathbf{Y}) = \mathcal{N}(\mathbf{X} \mid \mathbf{M}\mathbf{Y} + \mathbf{b}, \boldsymbol{\Sigma})$。该条件模型需要多少个独立参数？

**详细解答**：
- 线性变换矩阵 $\mathbf{M} \in \mathbb{R}^{2 \times 2}$ 包含 $2 \times 2 = 4$ 个参数；
- 偏置向量 $\mathbf{b} \in \mathbb{R}^2$ 包含 2 个参数；
- 协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{2 \times 2}$ 为对称正定矩阵，包含 $2 \times (2+1)/2 = 3$ 个参数；

总计共需 $4 + 2 + 3 = \mathbf{9}$ 个独立参数。

---

### 习题 2.7 (Exercise 2.7)
**题目**：给定一个包含 5 个节点的树状贝叶斯网络，其中每个节点均可取 4 个离散离散状态，每个子节点恰有 1 个父节点（除根节点外）。若不利用网络拓扑独立性，完全联合分布需要多少参数？利用该贝叶斯网络后参数量减少了百分之几？

**详细解答**：
1. **完全联合分布**：5 个 4 值变量共有 $4^5 = 1024$ 种组合，需要 $1024 - 1 = \mathbf{1023}$ 个独立参数；
2. **贝叶斯网络分解**：
   - 根节点需要 $4 - 1 = 3$ 个参数；
   - 其余 4 个子节点各自具有 1 个 4 值父节点，每个子节点的条件概率表有 4 行，每行包含 $4 - 1 = 3$ 个独立参数，单个子节点需 $4 \times 3 = 12$ 个参数；
   - 4 个子节点共需 $4 \times 12 = 48$ 个参数；
   - 全网总参数量为 $3 + 48 = \mathbf{51}$ 个参数；
3. **减少比例**：
   $$
   \frac{1023 - 51}{1023} \times 100\% = \frac{972}{1023} \approx \mathbf{95.01\%}
   $$
参数量大幅减少了 95% 以上。

---

### 习题 2.8 (Exercise 2.8)
**题目**：给定如下贝叶斯网络拓扑图，在给定观测集合 $\{C\}$ 条件下，变量 $A$ 是否与 $E$ 达成 d-分离，即是否满足 $(A \perp E \mid C)$？

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/exercise_bn.png" alt="习题 2.8 贝叶斯网络拓扑" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 2.13：习题 2.8 与 2.9 的贝叶斯网络拓扑图。</p>
</div>

**详细解答**：
连接 $A$ 与 $E$ 的无向路径有两条：
1. **路径一 $A \to D \to E$**：中间节点 $D \notin \{C\}$ 未被观测，因果链未被阻断；
2. **路径二 $A \to C \to E$**：中间节点 $C \in \{C\}$ 已被观测，因果链被阻断。

由于路径一保持畅通，因此变量 $A$ 与 $E$ **不满足 d-分离**，故 $(A \not\perp E \mid C)$。

---

### 习题 2.9 (Exercise 2.9)
**题目**：针对习题 2.8 中图 2.13 所示的网络拓扑，求变量节点 $B$ 的**马尔可夫毯（Markov Blanket）**。

**详细解答**：
在贝叶斯网络中，任意节点 $X$ 的马尔可夫毯 $\text{MB}(X)$ 由三部分组成：
1. $X$ 的所有父节点（Parents）；
2. $X$ 的所有子节点（Children）；
3. $X$ 的所有配偶节点（Spouses，即与 $X$ 共同拥有同一子节点的其他父节点）。

对于节点 $B$：
- 父节点：$A$；
- 子节点：$D$；
- 配偶节点：检查子节点 $D$ 的其他父节点，发现节点 $C$ 同样指向 $D$（倒置分叉 $B \to D \leftarrow C$），因此 $C$ 是 $B$ 的配偶节点；
- 综合得出，节点 $B$ 的马尔可夫毯为：
$$
\text{MB}(B) = \{A, C, D\}
$$
给定其马尔可夫毯中的所有变量时，节点 $B$ 与网络中所有其他其余节点条件独立。

---

### 习题 2.10 (Exercise 2.10)
**题目**：在一个具有直接有向边 $A \to B$ 的贝叶斯网络中，变量 $A$ 与变量 $B$ 是否有可能在数值上相互独立？

**详细解答**：
**有可能**。
有向边 $A \to B$ 仅表明网络结构**不蕴含（does not imply）**两者的独立性（即不能在任意参数下均保证独立）；但这绝不意味着在某些特殊的条件概率表数值赋值下它们不能独立。例如：若条件概率分布 $P(B \mid A)$ 满足对于 $A$ 的所有可能取值 $a$，都有 $P(B = b \mid A = a) = P(B = b)$ 恒成立，则变量 $A$ 与 $B$ 在数值上完全独立。图形结构表达的是通用的独立性保证，而非相依性的绝对强制。
