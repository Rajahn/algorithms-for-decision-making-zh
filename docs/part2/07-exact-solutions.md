# 第 7 章：精确解法 (Exact Solution Methods)

在第一部分中，我们探讨了单步简单决策问题。然而在绝大多数实际工程应用中，现实决策是**序贯发生（sequential）**的：智能体在当前时刻采取的行动，不仅产生即时回报，而且深刻改变了环境未来的物理演化状态，进而影响后续的所有可能决策。

本章正式引入序贯决策制定的标准数学基石——**马尔可夫决策过程（Markov Decision Processes, MDP）**。我们系统阐述完全可观测有限状态空间下的精确动态规划解法。首先形式化定义 MDP 及其平稳性质；随后详细推导状态价值函数的贝尔曼期望方程与矩阵解析求解算法；接着探讨动作价值函数（$Q$ 函数）与贪心策略改进；进而系统推导交替迭代的**策略迭代（Policy Iteration）**算法与基于压缩映射不动点原理的**价值迭代（Value Iteration）**算法；最后介绍异步价值迭代与对偶线性规划求解范式。

---

## 7.1 马尔可夫决策过程 (Markov Decision Processes)

在序贯决策中，环境的物理演化跨越多个连续的时间步 $t = 0, 1, 2, \dots$。如果智能体在做出决策时，环境的历史所有信息已经全部浓缩在**当前状态（Current State）**之中，即未来演化仅取决于当前状态与所选行动，而与过去的漫长历史无关，则称系统满足**马尔可夫性质（Markov Property）**。

### 7.1.1 形式化定义
一个离散时间的马尔可夫决策过程（MDP）由五元组 $(\mathcal{S}, \mathcal{A}, T, R, \gamma)$ 精确定义：
1. **状态空间 $\mathcal{S}$**：智能体所处环境的所有可能物理状态集合。若集合元素有限，记状态数为 $|\mathcal{S}|$；
2. **行动空间 $\mathcal{A}$**：智能体在各状态下可执行的决策行动集合；
3. **状态转移模型 $T(s' \mid s, a)$**：在状态 $s \in \mathcal{S}$ 下执行行动 $a \in \mathcal{A}$ 后，下一时刻转移至新状态 $s' \in \mathcal{S}$ 的条件概率分布，满足对任意 $s, a$ 均有 $\sum_{s' \in \mathcal{S}} T(s' \mid s, a) = 1$；
4. **奖励函数 $R(s, a)$**：智能体在状态 $s$ 下采取行动 $a$ 时所获得的标量即时反馈（数值效用回报）。有时奖励函数亦形式化为 $R(s, a, s')$，但数学上可通过对后继状态积分转化为期望即时奖励 $R(s, a) = \sum_{s'} T(s' \mid s, a) R(s, a, s')$；
5. **折扣因子 $\gamma \in [0, 1)$**：反映智能体对未来回报的时间偏好（时间价值贴现）。$\gamma$ 越接近 1，智能体越具有前瞻远见；$\gamma$ 越接近 0，智能体越聚焦于即时回报。在无限时域（Infinite Horizon）中，$\gamma < 1$ 严格保证了累积总回报级数的收敛性。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_7_1.png" alt="有限时域 MDP 展开图" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.1：有限时域 MDP 的决策网络展开图。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_7_2.png" alt="平稳 MDP 决策网络" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.2：平稳（Stationary）MDP 的单步递推决策网络图。</p>
  </div>
</div>

```julia
# MDP 核心数据结构定义 (来自官方 Julia 算法实现)
struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
    TR # sample transition and reward
end
####################
```

---

## 7.2 策略评估 (Policy Evaluation)

智能体的行为逻辑由**策略（Policy）$\pi(s)$** 形式化规范。对于确定性平稳策略，$\pi: \mathcal{S} \to \mathcal{A}$ 将每个可能的状态直接映射为一个确定的动作。

### 7.2.1 状态价值函数与贝尔曼期望方程
给定策略 $\pi$，从初始状态 $s_0 = s$ 出发，遵循该策略执行无限期交互所获得的**未来累积贴现总回报的期望值**，定义为该策略的**状态价值函数（Value Function / Utility）** $U^\pi(s)$：
$$
U^\pi(s) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, \pi(s_t)) \;\middle|\; s_0 = s, \; s_{t+1} \sim T(\cdot \mid s_t, \pi(s_t)) \right]
$$
展开第一项即时回报，可将无穷级数递推分解为著名的**贝尔曼期望方程（Bellman Expectation Equation）**：
$$
U^\pi(s) = R(s, \pi(s)) + \gamma \sum_{s' \in \mathcal{S}} T(s' \mid s, \pi(s)) U^\pi(s')
$$
该方程表明：任意状态的价值，恒等于**当前执行动作的即时奖励**，加上**下一后继状态期望折现价值**之和。

### 7.2.2 矩阵解析解法
对于包含有限个状态的状态空间，贝尔曼期望方程实质上是一个包含 $|\mathcal{S}|$ 个未知数与 $|\mathcal{S}|$ 个方程的线性方程组。
定义列向量 $\mathbf{U}^\pi \in \mathbb{R}^{|\mathcal{S}|}$、即时奖励向量 $\mathbf{R}^\pi \in \mathbb{R}^{|\mathcal{S}|}$ 以及状态转移矩阵 $\mathbf{T}^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$，其中第 $i$ 行第 $j$ 列元素为 $T(s_j \mid s_i, \pi(s_i))$。
方程组可紧凑写为矩阵形式：
$$
\mathbf{U}^\pi = \mathbf{R}^\pi + \gamma \mathbf{T}^\pi \mathbf{U}^\pi \implies (\mathbf{I} - \gamma \mathbf{T}^\pi) \mathbf{U}^\pi = \mathbf{R}^\pi
$$
由于 $\gamma < 1$ 且转移矩阵行和为 1，矩阵 $(\mathbf{I} - \gamma \mathbf{T}^\pi)$ 严格可逆。因此存在唯一的解析闭式解：
$$
\mathbf{U}^\pi = (\mathbf{I} - \gamma \mathbf{T}^\pi)^{-1} \mathbf{R}^\pi
$$
其计算时间复杂度为高斯消元求逆的 $O(|\mathcal{S}|^3)$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_7_4.png" alt="六边形世界策略精确评估" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.3：在六边形网格世界（Hex World）中，对始终朝东移动的固定策略进行矩阵解析评估所得的精确状态效用热力分布。</p>
</div>

### 7.2.3 迭代式策略评估 (Iterative Policy Evaluation)
当状态空间规模较大时，求逆矩阵开销巨大。我们可以利用动态规划从任意初始猜测值 $U^{(0)}$ 出发，持续将贝尔曼期望算子作为更新规则：
$$
U^{(k+1)}(s) \leftarrow R(s, \pi(s)) + \gamma \sum_{s'} T(s' \mid s, \pi(s)) U^{(k)}(s')
$$
由于贝尔曼算子具有 $\gamma$-压缩性，迭代序列保证以几何速率单调线性收敛至真实解 $\mathbf{U}^\pi$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_7_3.png" alt="迭代式策略评估过程" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.4：迭代式策略评估在六边形世界中随着迭代轮数递增（第 1 轮、第 2 轮直至收敛）的状态价值演化全景。</p>
</div>

```julia
# 策略评估算法实现
function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end
function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end
####################
```

---

## 7.3 动作价值函数与策略改进 (Policy Improvement)

在评估出某策略的价值函数 $U^\pi(s)$ 之后，我们如何改进该策略以获取更高回报？

定义**动作价值函数（Action-Value Function / $Q$ 函数）** $Q(s, a)$ 为：在状态 $s$ 下首先强制执行动作 $a$，此后严格遵循既有策略 $\pi$ 推进所能获得的期望总回报：
$$
Q(s, a) = R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) U(s')
$$
根据最大期望效用准则，我们在每一个状态下贪心地选择能够使动作价值达到最大的动作，构造一个全新的策略 $\pi'$：
$$
\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a) = \arg\max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) U(s') \right]
$$

**策略改进定理（Policy Improvement Theorem）**：由上述贪心方式生成的新策略 $\pi'$，在全状态空间上严格保证不劣于原策略：
$$
U^{\pi'}(s) \ge U^\pi(s) \quad (\forall s \in \mathcal{S})
$$
如果两者的效用函数在所有状态上完全相等，则表明当前策略已经达到全局**最优策略（Optimal Policy）** $\pi^*$。

---

## 7.4 策略迭代 (Policy Iteration)

结合上述两个步骤，罗纳德·霍华德（Ronald Howard, 1960）提出了求解最优策略的**策略迭代（Policy Iteration）**算法。

### 算法循环架构：
1. **策略评估（Policy Evaluation）**：给定当前策略 $\pi^{(k)}$，精确求解线性方程组计算其真实价值函数 $U^{\pi^{(k)}}$；
2. **策略改进（Policy Improvement）**：根据 $U^{\pi^{(k)}}$ 贪心更新策略，得到新策略 $\pi^{(k+1)}$；
3. **终止条件**：若 $\pi^{(k+1)}(s) = \pi^{(k)}(s)$ 对所有状态 $s$ 均成立，算法宣告收敛并输出最优策略 $\pi^*$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_7_5.png" alt="策略迭代收敛过程" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.5：策略迭代在六边形世界中的收敛过程。从初始随机策略（第 1 轮）到单调改进（第 2 轮），通常在极少数轮次内即可达到严格最优策略。</p>
</div>

由于有限状态有限动作下的确定性策略总数最多为 $|\mathcal{A}|^{|\mathcal{S}|}$ 个，且每轮迭代的策略价值均严格单调递增，策略绝不可能重复出现。因此**策略迭代算法必然在有限步内严格收敛至全局最优策略**。

```julia
# 策略迭代算法实现
function iterative_policy_evaluation(𝒫::MDP, π, k_max)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    U = [0.0 for s in 𝒮]
    for k in 1:k_max
        U = [lookahead(𝒫, U, s, π(s)) for s in 𝒮]
    end
    return U
end
####################
```

---

## 7.5 价值迭代 (Value Iteration)

策略迭代在每轮外层循环中都需要精确求解一次全局线性方程组，在大规模状态空间下单步耗时较长。贝尔曼（Richard Bellman, 1957）提出了将优化步骤直接融入价值更新的**价值迭代（Value Iteration）**算法。

### 7.5.1 贝尔曼最优性方程与压缩映射定理
全局最优策略的价值函数 $U^*(s)$ 必定满足**贝尔曼最优性方程（Bellman Optimality Equation）**：
$$
U^*(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} T(s' \mid s, a) U^*(s') \right]
$$
定义**贝尔曼最优性算子 $\mathcal{B}: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$**：
$$
(\mathcal{B} U)(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) U(s') \right]
$$

**核心数学定理（巴拿赫不动点定理与 $\gamma$-压缩映射）**：
对于定义在无穷范数（$\|U\|_\infty = \max_s |U(s)|$）上的价值空间，贝尔曼最优性算子 $\mathcal{B}$ 是一个**严格收缩率等于折扣因子 $\gamma$ 的压缩映射**：
$$
\|\mathcal{B} U_1 - \mathcal{B} U_2\|_\infty \le \gamma \|U_1 - U_2\|_\infty
$$
根据完备度量空间上的巴拿赫不动点定理，无论从何种初始猜测 $U^{(0)}$ 出发，反复应用算子更新 $U^{(k+1)} = \mathcal{B} U^{(k)}$，迭代序列**必然几何级数收敛至唯一的全局最优不动点 $U^*$**！

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_7_6.png" alt="价值迭代在 Hex World 中的演化" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 7.6：价值迭代算法在 Hex World 中的演化切片。从初始 $U^{(0)}=0$ 出发，奖励信号逐步由目标终点向整个网格平滑回传，最终收敛至全局最优价值曲面。</p>
</div>

### 7.5.2 终止准则
在数值计算中，我们不可能执行无穷次迭代。当相邻两次迭代的最大状态变动量满足以下界限时：
$$
\|U^{(k+1)} - U^{(k)}\|_\infty < \epsilon \frac{1 - \gamma}{2\gamma}
$$
可以严格从数学上证明，基于当前价值函数 $U^{(k+1)}$ 贪心提取出的策略，与真实全局最优策略之间的效用损失必然小于 $\epsilon$：
$$
\|U^{\pi_{k+1}} - U^*\|_\infty < \epsilon
$$

```julia
# 价值迭代核心算法实现
function policy_evaluation(𝒫::MDP, π)
	𝒮, R, T, γ = 𝒫.𝒮, 𝒫.R, 𝒫.T, 𝒫.γ
	R′ = [R(s, π(s)) for s in 𝒮]
	T′ = [T(s, π(s), s′) for s in 𝒮, s′ in 𝒮]
	return (I - γ*T′)\R′
end
####################
```

---

## 7.6 异步价值迭代与优先遍历 (Asynchronous Value Iteration)

标准价值迭代每次遍历都必须在内存中同步创建一份完整的副本以更新所有状态。为了提升缓存局部性与计算效率，通常采用**异步更新技术**：

1. **高斯-赛德尔价值迭代（Gauss-Seidel Value Iteration）**：直接在原数组上原地（in-place）更新状态效用，后更新的状态能够立即使用先更新状态的最新数值，收敛速度明显加快；
2. **优先遍历（Prioritized Sweeping）**：维护一个基于**贝尔曼误差残差（Bellman Error）**的优先队列，每次优先弹出并更新那些因后继状态发生大幅变动而产生最剧烈冲击的状态节点，将宝贵算力精准聚焦在关键瓶颈状态。

---

## 7.7 线性规划解法 (Linear Programming Formulation)

除了动态规划，求解最优价值函数还可以被严密等价形式化为一个**线性规划（Linear Programming, LP）**问题。

根据贝尔曼最优性方程，最优价值函数 $U^*$ 是满足以下一组不等式约束的逐点最小下界：
$$
U(s) \ge R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) U(s') \quad (\forall s \in \mathcal{S}, \; \forall a \in \mathcal{A})
$$
因此，寻找最优价值函数等价于求解如下主问题：
$$
\begin{aligned}
\min_{\mathbf{U}} \quad & \sum_{s \in \mathcal{S}} U(s) \\
\text{s.t.} \quad & U(s) - \gamma \sum_{s'} T(s' \mid s, a) U(s') \ge R(s, a) \quad (\forall s \in \mathcal{S}, \; \forall a \in \mathcal{A})
\end{aligned}
$$
该线性规划包含 $|\mathcal{S}|$ 个优化变量和 $|\mathcal{S}| \times |\mathcal{A}|$ 个不等式约束，可利用标准的内点法或单纯形法直接求得全局数值精确解。

---

## 7.8 本章小结 (Summary)

- **序贯决策基石**：MDP 通过引入状态空间、动作空间、转移模型、即时奖励与折扣因子，将单步决策拓展至无限时域因果演化；
- **贝尔曼期望方程**：刻画了固定策略价值函数关于后继状态的递归平衡关系，在有限状态下具有矩阵求逆的封闭解析解；
- **策略迭代的有限步收敛**：通过交替执行精确策略评估与贪心策略改进，在单调递增保证下以超线性速度在有限轮内逼近最优策略；
- **价值迭代的不动点保证**：贝尔曼最优性算子在无穷范数下具有严格的 $\gamma$-压缩收敛性，通过设定合理的残差阈值可获得有保证的近似最优解；
- **优化对偶性**：贝尔曼方程亦可转化为包含 $|\mathcal{S}| \times |\mathcal{A}|$ 条约束的线性规划问题求解。

---

## 7.9 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 7.1 (Exercise 7.1)
**题目**：考虑一个两状态 MDP，$\mathcal{S} = \{s_1, s_2\}$，仅有一个固定动作。已知转移概率矩阵与即时奖励向量为：
$$
\mathbf{T} = \begin{bmatrix} 0.7 & 0.3 \\ 0.2 & 0.8 \end{bmatrix}, \quad \mathbf{R} = \begin{bmatrix} 5 \\ 10 \end{bmatrix}, \quad \gamma = 0.9
$$
利用矩阵解析解法求解该平稳策略的状态效用向量 $\mathbf{U} = [U(s_1), U(s_2)]^\top$。

**详细解答**：
代入矩阵方程 $(\mathbf{I} - \gamma \mathbf{T}) \mathbf{U} = \mathbf{R}$：
$$
\mathbf{I} - \gamma \mathbf{T} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - 0.9 \begin{bmatrix} 0.7 & 0.3 \\ 0.2 & 0.8 \end{bmatrix} = \begin{bmatrix} 1 - 0.63 & -0.27 \\ -0.18 & 1 - 0.72 \end{bmatrix} = \begin{bmatrix} 0.37 & -0.27 \\ -0.18 & 0.28 \end{bmatrix}
$$
计算行列式：
$$
\det = 0.37 \times 0.28 - (-0.27) \times (-0.18) = 0.1036 - 0.0486 = 0.055
$$
求逆矩阵：
$$
(\mathbf{I} - \gamma \mathbf{T})^{-1} = \frac{1}{0.055} \begin{bmatrix} 0.28 & 0.27 \\ 0.18 & 0.37 \end{bmatrix}
$$
相乘求解效用：
$$
\mathbf{U} = \frac{1}{0.055} \begin{bmatrix} 0.28 \times 5 + 0.27 \times 10 \\ 0.18 \times 5 + 0.37 \times 10 \end{bmatrix} = \frac{1}{0.055} \begin{bmatrix} 1.4 + 2.7 \\ 0.9 + 3.7 \end{bmatrix} = \frac{1}{0.055} \begin{bmatrix} 4.1 \\ 4.6 \end{bmatrix} \approx \begin{bmatrix} \mathbf{74.55} \\ \mathbf{83.64} \end{bmatrix}
$$

---

### 习题 7.2 (Exercise 7.2)
**题目**：证明贝尔曼期望算子 $\mathcal{T}^\pi: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$ 是一个 $\gamma$-压缩映射，即对于任意两组价值向量 $\mathbf{U}$ 与 $\mathbf{V}$，均满足 $\|\mathcal{T}^\pi \mathbf{U} - \mathcal{T}^\pi \mathbf{V}\|_\infty \le \gamma \|\mathbf{U} - \mathbf{V}\|_\infty$。

**详细解答**：
对于任意状态 $s$：
$$
\begin{aligned}
|(\mathcal{T}^\pi \mathbf{U})(s) - (\mathcal{T}^\pi \mathbf{V})(s)| &= \left| \left( R(s, \pi(s)) + \gamma \sum_{s'} T(s' \mid s, \pi(s)) U(s') \right) - \left( R(s, \pi(s)) + \gamma \sum_{s'} T(s' \mid s, \pi(s)) V(s') \right) \right| \\
&= \gamma \left| \sum_{s'} T(s' \mid s, \pi(s)) (U(s') - V(s')) \right| \\
&\le \gamma \sum_{s'} T(s' \mid s, \pi(s)) |U(s') - V(s')| \quad (\text{绝对值三角不等式}) \\
&\le \gamma \sum_{s'} T(s' \mid s, \pi(s)) \max_{s''} |U(s'') - V(s'')| \\
&= \gamma \| \mathbf{U} - \mathbf{V} \|_\infty \sum_{s'} T(s' \mid s, \pi(s)) \\
&= \gamma \| \mathbf{U} - \mathbf{V} \|_\infty \quad (\text{因概率分布和为 } 1)
\end{aligned}
$$
由于该不等式对所有状态 $s \in \mathcal{S}$ 均成立，两边关于 $s$ 取上确界即可得：
$$
\|\mathcal{T}^\pi \mathbf{U} - \mathcal{T}^\pi \mathbf{V}\|_\infty \le \gamma \|\mathbf{U} - \mathbf{V}\|_\infty
$$
证毕。

---

### 习题 7.3 (Exercise 7.3)
**题目**：在价值迭代中，假设折扣因子 $\gamma = 0.95$。若我们希望最终提取出的贪心策略与真实最优策略之间的价值误差严格控制在 $\epsilon = 0.1$ 之内，相邻两轮迭代的残差阈值应当设置为何值？

**详细解答**：
代入价值迭代的终止阈值公式：
$$
\text{threshold} = \epsilon \frac{1 - \gamma}{2\gamma} = 0.1 \times \frac{1 - 0.95}{2 \times 0.95} = 0.1 \times \frac{0.05}{1.9} = \frac{0.005}{1.9} \approx \mathbf{0.00263}
$$
故只要相邻两轮全状态最大变动量小于约 $0.00263$，即可保证策略误差上限不超过 0.1。

---

### 习题 7.4 (Exercise 7.4)
**题目**：若将所有状态的即时奖励同时增加一个常数 $c$，即新奖励为 $\tilde{R}(s, a) = R(s, a) + c$。求最优策略 $\pi^*$ 是否会发生改变？新的最优价值函数与原最优价值函数之间有何数学关系？

**详细解答**：
设原最优价值函数为 $U^*(s)$。考虑将常数 $c$ 引入贴现求和：
$$
\tilde{U}^\pi(s) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t (R(s_t, a_t) + c) \right] = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right] + c \sum_{t=0}^\infty \gamma^t = U^\pi(s) + \frac{c}{1 - \gamma}
$$
因此对于任意策略 $\pi$，其新价值函数恒等于原价值函数加上常数 $\frac{c}{1 - \gamma}$。
在进行策略贪心选择时：
$$
\arg\max_a \left[ \tilde{R}(s, a) + \gamma \sum_{s'} T(s' \mid s, a) \tilde{U}(s') \right] = \arg\max_a \left[ R(s, a) + c + \gamma \sum_{s'} T(s' \mid s, a) \left( U(s') + \frac{c}{1 - \gamma} \right) \right]
$$
括号内相较于原项仅增加了常数 $c + \gamma \frac{c}{1 - \gamma} = \frac{c}{1 - \gamma}$，该常数与动作 $a$ 完全无关。
因此，**最优策略 $\pi^*$ 完全保持不变**，新的最优价值函数处处平移 $\mathbf{U}^*(s) + \frac{c}{1 - \gamma}$。

---

### 习题 7.5 (Exercise 7.5)
**题目**：考虑一个单状态、两动作的简单 MDP，动作集合 $\mathcal{A} = \{a_1, a_2\}$。自环转移概率均为 1。已知奖励为 $R(s, a_1) = 2$，$R(s, a_2) = 3$。折扣因子 $\gamma = 0.8$。从初始猜测 $U^{(0)}(s) = 0$ 出发，写出前 3 轮价值迭代的具体数值。

**详细解答**：
贝尔曼最优更新算子为：$U^{(k+1)}(s) = \max_{a \in \{a_1, a_2\}} [ R(s, a) + \gamma U^{(k)}(s) ]$。
显然动作 $a_2$ 的奖励 $3 > 2$ 始终占优，故递推式简化为 $U^{(k+1)} = 3 + 0.8 U^{(k)}$：
- 第 1 轮：$U^{(1)} = 3 + 0.8 \times 0 = \mathbf{3.0}$；
- 第 2 轮：$U^{(2)} = 3 + 0.8 \times 3.0 = 3 + 2.4 = \mathbf{5.4}$；
- 第 3 轮：$U^{(3)} = 3 + 0.8 \times 5.4 = 3 + 4.32 = \mathbf{7.32}$；
（解析极限值为 $\frac{3}{1 - 0.8} = 15.0$）。

---

### 习题 7.6 (Exercise 7.6)
**题目**：在策略迭代中，为什么策略改进步骤保证不会产生性能比上一轮更差的新策略？简述其推导逻辑。

**详细解答**：
设新策略为 $\pi'(s) = \arg\max_a Q^{\pi}(s, a)$。根据定义，对于任意状态 $s$ 均有：
$$
Q^\pi(s, \pi'(s)) = \max_a Q^\pi(s, a) \ge Q^\pi(s, \pi(s)) = U^\pi(s)
$$
反复将该不等式代入动作价值展开式中展开：
$$
U^\pi(s) \le Q^\pi(s, \pi'(s)) = R(s, \pi'(s)) + \gamma \sum_{s'} T(s' \mid s, \pi'(s)) U^\pi(s')
$$
将右侧的 $U^\pi(s')$ 递归用 $Q^\pi(s', \pi'(s'))$ 继续放大，经过无限次迭代展开后，右端严格收敛至新策略的累积贴现价值 $U^{\pi'}(s)$。
由此严格证明 $U^{\pi'}(s) \ge U^\pi(s)$ 恒成立。

---

### 习题 7.7 (Exercise 7.7)
**题目**：在什么情况下，策略迭代会比价值迭代具有压倒性的收敛速度优势？

**详细解答**：
当**折扣因子 $\gamma$ 非常接近 1**（例如 $\gamma = 0.999$）时，价值迭代的收敛因子 $(1 - \gamma)$ 极小，价值信号在网格中的反向传播极其缓慢，往往需要数千次迭代才能收敛。而策略迭代每轮直接利用矩阵逆求解全局不动点，跨越了局部传播的限制，并且策略空间是有限的离散集合，往往只需十余轮策略交替即可找到全局最优策略。

---

### 习题 7.8 (Exercise 7.8)
**题目**：高斯-赛德尔异步价值迭代在更新当前状态 $s_i$ 时，与同步价值迭代有何根本区别？

**详细解答**：
- **同步价值迭代**：在第 $k+1$ 轮迭代计算所有状态的新价值时，右侧项统一强制读取上一轮缓存的静态数组 $U^{(k)}(s')$，更新完毕后才全量替换；
- **高斯-赛德尔异步迭代**：仅维护一个单一的全局共享价值数组。在更新状态 $s_i$ 时，后继状态中那些在当前轮次排在 $s_i$ 之前的状态已经更新到了第 $k+1$ 轮的最新数值，更新立即生效，从而加速了价值信息的单向流动。

---

### 习题 7.9 (Exercise 7.9)
**题目**：考虑一个包含两个等价最优动作的 MDP，若策略迭代在评估时在两个动作间无规则随机震荡，是否会导致死循环？如何避免？

**详细解答**：
若存在多个动作使得 $Q(s, a)$ 达到相同的极大值，如果打破并列的规则不固定（例如随机选择），算法可能在两个产生相同最高效用的策略之间反复横跳。
**避免方法**：在策略改进选择动作时施加**确定性打破规则（Deterministic Tie-Breaking Rule）**，例如固定选择动作索引较小的动作。由于动作空间和策略空间有限且价值严格单调非减，确定性打破规则可严格保证策略迭代绝不陷入死循环。
