# 第 23 章：控制器抽象 (Controller Abstractions)

在前面几章中，我们要么将策略表示为高维单纯形空间上的分段线性凸函数（Alpha 向量集合），要么表示为庞大的在线前瞻树。然而在资源极端受限的嵌入式微处理器（如微型无人机飞控芯片、微型心脏起搏器）上，既没有足够的内存存储数百个 Alpha 向量，也没有足够的算力运行在线蒙特卡洛树搜索。

**有限状态控制器（Finite State Controllers, FSC）**提供了一种兼具极低运行开销与时序记忆能力的紧凑策略抽象范式。它将智能体的策略直接编译为一个固定规模的**有限状态机（Finite State Machine）**。智能体在执行动作的同时更新控制器的内部离散记忆节点。

本章系统探讨控制器抽象与有限状态机策略理论。我们首先形式化定义有限状态控制器的图拓扑与转移映射；随后深入推导控制器策略在 POMDP 环境下的**闭环对偶马尔可夫链稳态评估方程**；最后探讨基于非线性规划与梯度上升的控制器参数直接优化算法。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（把策略固化为芯片状态机）**：在功耗极低的微型无人机芯片或起搏器上，既没有内存存几千个 Alpha 向量，也没有算力跑 MCTS。有限状态控制器（FSC）将策略直接编译为一个固定规模的**有限状态机（Finite State Machine）**。
- **优雅的交叉积马尔可夫链**：
  - 智能体当前的大脑只有有限个离散记忆节点 $x \in \{x_1, \dots, x_n\}$；
  - 在节点 $x$ 执行绑定动作，根据接收到的传感器观测 $o$ 顺着连线跳到新节点 $x'$；
  - 物理环境状态 $s$ 与记忆节点 $x$ 的复合状态 $(s, x)$ 构成了一个尺寸仅为 $|\mathcal{S}| \times n$ 的闭环标准马尔可夫链；
  - 求解其稳态累积价值只需做一次简单的矩阵求逆！
:::
---

## 23.1 有限状态控制器形式化 (Finite State Controllers)

一个有限状态控制器由一个有限大小的**内部记忆节点集合 $\mathcal{X} = \{x_1, \dots, x_n\}$** 构成：
- 每个内部节点 $x \in \mathcal{X}$ 对应一个确定性或随机性的动作选择策略 $\psi(a \mid x) = P(A = a \mid x)$；
- 当智能体执行动作 $a$ 并从物理环境接收到传感器观测 $o \in \mathcal{O}$ 时，控制器根据内部转移规则跳转至新的记忆状态：
  $$
  x' \sim \eta(x' \mid x, a, o)
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_23_1.png" alt="有限状态控制器图拓扑结构" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 23.1：一个包含 4 个离散记忆节点的有限状态控制器拓扑。节点标注执行的物理动作，有向转移边由环境反馈的观测标签所触发。</p>
</div>

```julia
# 有限状态控制器数据结构定义 (来自官方 Julia 算法实现)
mutable struct ControllerPolicy
    𝒫 # problem
    X # set of controller nodes
    ψ # action selection distribution
    η # successor selection distribution
end

function (π::ControllerPolicy)(x)
    𝒜, ψ = π.𝒫.𝒜, π.ψ
    dist = [ψ[x, a] for a in 𝒜]
    return rand(SetCategorical(𝒜, dist))
end

function update(π::ControllerPolicy, x, a, o)
    X, η = π.X, π.η
    dist = [η[x, a, o, x′] for x′ in X]
    return rand(SetCategorical(X, dist))
end
####################
```

### 极致的工程执行效率
一旦控制器参数离线训练完成，在线执行仅需极简的常数时间：
1. 读取当前节点绑定的动作并执行（查表 $O(1)$）；
2. 接收传感器观测 $o$，沿对应观测出边跳转至下一个节点（指针跳转 $O(1)$）；
3. **内存开销仅为数个到几十个整型节点**，完全无需维护连续置信分布或运行任何概率滤波算法！

---

## 23.2 控制器策略评估 (Policy Evaluation for Controllers)

当一个包含 $n$ 个记忆节点的有限状态控制器与一个包含 $|\mathcal{S}|$ 个物理状态的 POMDP 闭环耦合时，其整体复合系统构成了一个**状态空间大小为 $|\mathcal{S}| \times n$ 的标准完全可观测马尔可夫链（Cross-Product Markov Chain）**！

定义复合状态为元组 $(s, x) \in \mathcal{S} \times \mathcal{X}$。
复合状态转移概率由物理转移模型与控制器跳转模型严密结合给出：
$$
P(s', x' \mid s, x) = \sum_{a \in \mathcal{A}} \psi(a \mid x) T(s' \mid s, a) \sum_{o \in \mathcal{O}} O(o \mid a, s') \eta(x' \mid x, a, o)
$$
即时奖励为：
$$
R(s, x) = \sum_{a \in \mathcal{A}} \psi(a \mid x) R(s, a)
$$

### 线性方程组闭式解：
复合状态的价值函数 $U(s, x)$ 严格满足贝尔曼期望方程：
$$
U(s, x) = R(s, x) + \gamma \sum_{s' \in \mathcal{S}} \sum_{x' \in \mathcal{X}} P(s', x' \mid s, x) U(s', x')
$$
由于这是一个维度仅为 $|\mathcal{S}| \cdot n$ 的标准线性方程组，可直接调用第 7 章的矩阵求逆在多项式时间内精确求解出控制器在所有物理状态下的确切累积贴现回报！

```julia
# 有限状态控制器精确评估算法实现
function utility(π::ControllerPolicy, U, x, s)
    𝒮, 𝒜, 𝒪 = π.𝒫.𝒮, π.𝒫.𝒜, π.𝒫.𝒪
    T, O, R, γ = π.𝒫.T, π.𝒫.O, π.𝒫.R, π.𝒫.γ
    X, ψ, η = π.X, π.ψ, π.η
    U′(a,s′,o) = sum(η[x,a,o,x′]*U[x′,s′] for x′ in X)
    U′(a,s′) = T(s,a,s′)*sum(O(a,s′,o)*U′(a,s′,o) for o in 𝒪)
    U′(a) = R(s,a) + γ*sum(U′(a,s′) for s′ in 𝒮)
    return sum(ψ[x,a]*U′(a) for a in 𝒜)
end

function iterative_policy_evaluation(π::ControllerPolicy, k_max)
    𝒮, X = π.𝒫.𝒮, π.X
    U = Dict((x, s) => 0.0 for x in X, s in 𝒮)
    for k in 1:k_max
        U = Dict((x, s) => utility(π, U, x, s) for x in X, s in 𝒮)
    end
    return U
end
####################
```

---

---

## 23.3 非线性规划优化控制器 (Nonlinear Programming for Controllers)

除了局部爬山搜索，阿马托等人（Christopher Amato et al., 2010）证明：寻找给定规模的有限状态控制器的最优参数，可以被严密等价形式化为一个**非线性规划（Nonlinear Programming, NLP）**问题。

### 优化变量体系：
- 价值变量 $U(x, s)$：在内部记忆状态 $x$ 与物理状态 $s$ 下的复合价值；
- 策略变量 $\psi(a \mid x)$：内部节点到动作的输出概率；
- 转移变量 $\eta(x' \mid x, a, o)$：观测驱动的节点跳转概率。

### NLP 数学规划命题：
$$
\begin{aligned}
\max_{\mathbf{U}, \boldsymbol{\psi}, \boldsymbol{\eta}} \quad & \sum_{s \in \mathcal{S}} b_0(s) U(x_0, s) \\
\text{s.t.} \quad & U(x, s) \le \sum_{a \in \mathcal{A}} \psi(a \mid x) \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \sum_{o \in \mathcal{O}} O(o \mid a, s') \sum_{x'} \eta(x' \mid x, a, o) U(x', s') \right] \\
& \sum_{a \in \mathcal{A}} \psi(a \mid x) = 1, \quad \psi(a \mid x) \ge 0 \quad (\forall x, a) \\
& \sum_{x' \in \mathcal{X}} \eta(x' \mid x, a, o) = 1, \quad \eta(x' \mid x, a, o) \ge 0 \quad (\forall x, a, o)
\end{aligned}
$$
约束条件中包含了策略变量 $\psi$ 与转移变量 $\eta$ 以及价值变量 $U$ 之间的**三次连续乘积非线性项**。利用通用的非线性规划求解器（如 IPOPT 内点法或 SNOPT 顺序二次规划 SQP），算法能够直接在连续概率参数空间中进行全局或高质量局部收敛求解。

---

## 23.4 控制器参数优化与梯度上升

为了寻找最优的控制器参数（即动作输出概率 $\psi$ 与节点转移概率 $\eta$），通常采用以下方法：
1. **梯度上升法（Gradient Ascent on Controllers）**：将动作策略与转移概率参数化为 Softmax 可微形式，利用第 11 章的似然对数导数技巧推导复合马尔可夫链关于控制器参数的解析梯度，沿梯度上升优化；
2. **动态规划剪枝与节点合并**：类似于 Alpha 向量剪枝，若控制器中某两个记忆节点在所有物理状态下的价值曲面完全等价或被支配，可将冗余节点合并剔除，实现控制器的极度紧凑轻量化。

---

## 23.5 本章小结 (Summary)

- **终极紧凑的工程抽象**：有限状态控制器将历史时序记忆编译为离散有限状态机，消除了在线概率滤波的巨量开销；
- **交叉积马尔可夫链**：环境物理状态与控制器记忆节点的复合在代数上严格构成一个闭环标准马尔可夫链，使得全局策略评估具有精确线性的矩阵解析解；
- **端侧微控制器的守护神**：FSC 使得复杂的 POMDP 策略能够在功耗极低、算力孱弱的嵌入式芯片上实现硬实时执行。

---

## 23.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 23.1 (Exercise 23.1)
**题目**：考虑一个环境包含 2 个物理状态 $\mathcal{S} = \{s_1, s_2\}$，控制器包含 2 个内部记忆节点 $\mathcal{X} = \{x_1, x_2\}$。求其闭环交叉乘积马尔可夫链的全局状态转移矩阵 $\mathbf{P}$ 的维度大小。

**详细解答**：
复合状态空间由物理状态与内部节点的笛卡尔积构成：$\mathcal{S} \times \mathcal{X} = \{(s_1, x_1), (s_1, x_2), (s_2, x_1), (s_2, x_2)\}$。
状态总数为 $|\mathcal{S}| \times |\mathcal{X}| = 2 \times 2 = 4$ 个。
因此，对应的全局转移矩阵 $\mathbf{P}$ 是一个 **$4 \times 4$ 的方阵**。

---

### 习题 23.2 (Exercise 23.2)
**题目**：若将一个有限状态控制器的记忆节点数限制为 $n = 1$（仅包含单个记忆节点），该控制器退化为什么类型的控制策略？

**详细解答**：
当 $n = 1$ 时，控制器没有多余的内部状态可供跳转记忆，其转移规则退化为自环。此时该单一节点在每个时间步只能输出一个固定的静态动作选择概率分布 $\psi(a \mid x_1)$，其行为完全独立于过去的历史观测。因此该控制器严格退化为**纯无记忆的随机反应式静态策略（Memoryless / Reactive Policy）**。

---

### 习题 23.3 (Exercise 23.3)
**题目**：简述为什么有限状态控制器的参数优化问题在数学上是一个“非凸优化”（Non-Convex Optimization）难题。

**详细解答**：
因为改变控制器的参数（动作输出 $\psi$ 与转移核 $\eta$）会**同时非线性改变闭环系统的稳态状态占用分布测度与单步即时期望奖励**。两者在贝尔曼递归展开中表现为高阶多项式甚至有理分式复合，参数空间充斥着大量的鞍点与局部极小值，因而目标函数在全参数空间是非凸的。
