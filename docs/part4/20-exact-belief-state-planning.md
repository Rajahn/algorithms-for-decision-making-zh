# 第 20 章：精确置信状态规划 (Exact Belief State Planning)

在第 19 章中，我们证明了置信状态 $\mathbf{b}$ 是历史所有观测与行动的充分统计量。这意味着，**原本非马尔可夫的部分可观测系统，在数学上可以被严密等价转化为一个定义在连续单纯形置信空间上的全观测 MDP，称为置信 MDP（Belief MDP）**。

然而，置信 MDP 的状态空间是**连续的、多维的概率单纯形**，我们无法再直接使用离散表格型动态规划。索恩迪克（Richard Smallwood & Edward Sondik, 1971）取得了理论上的重大突破：他证明了**有限时域 POMDP 的最优价值函数不仅是连续的，而且在单纯形几何上严格表现为分段线性凸函数（Piecewise Linear and Convex, PWLC）**！

本章系统探讨有限时域与无限时域 POMDP 的**精确规划（Exact Planning）**理论。我们首先引入**条件计划树（Conditional Plans）**；随后严格推导**Alpha 向量（Alpha Vectors）**与超平面集合的数学对应关系；接着探讨通过线性规划消除被支配超平面的**精确剪枝（Pruning）**技术；最后系统阐述经典精确价值迭代算法（Monahan 算法）的跨步组合与收敛机理。

---

## 20.1 条件计划树 (Conditional Plans)

在完全可观测 MDP 中，策略是状态到动作的静态映射 $\pi(s)$。然而在 POMDP 中，智能体未来采取何种动作，取决于其未来将观察到何种具体的传感器读数。

因此，一个跨越 $k$ 个时间步的确定性决策方案，必须形式化为一棵**条件计划树（Conditional Plan Tree）** $\sigma$：
- 根节点指定当前时刻执行的第一个初始动作 $a$；
- 随后针对每一个可能返回的传感器观测结果 $o \in \mathcal{O}$，分别引出一条分支，指向一个深为 $k-1$ 步的子条件计划树 $\sigma(o)$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_20_1.png" alt="3步条件计划树结构" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 20.1：一棵 3 步有限时域条件计划树。节点标定执行的物理动作，边对应传感器返回的可能观测。</p>
</div>

### 组合爆炸的残酷性
在包含 $|\mathcal{A}|$ 个动作和 $|\mathcal{O}|$ 个观测的 POMDP 中：
- 1 步条件计划数有 $|\mathcal{A}|$ 个；
- 2 步条件计划数暴增至 $|\mathcal{A}| \cdot |\mathcal{A}|^{|\mathcal{O}|}$ 个；
- $k$ 步条件计划数遵循超指数双重递归增长：
  $$
  |\mathcal{P}_k| = |\mathcal{A}| \cdot |\mathcal{P}_{k-1}|^{|\mathcal{O}|}
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_20_2.png" alt="条件计划数量随步数双重指数增长" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 20.2：即使对于只有 2 个动作和 2 个观测的微型系统，条件计划数量也会在 4 步之内暴增至天文数字。</p>
</div>

---

## 20.2 Alpha 向量与分段线性凸函数 (Alpha Vectors & PWLC)

每一个特定的 $k$ 步条件计划 $\sigma$，其在任意底层物理状态 $s$ 下的期望贴现回报，由一个固定向量 $\boldsymbol{\alpha}_\sigma \in \mathbb{R}^{|\mathcal{S}|}$ 完全量化，该向量被称为 **Alpha 向量（Alpha Vector）**：
$$
\alpha_\sigma(s) = R(s, a_\sigma) + \gamma \sum_{s' \in \mathcal{S}} T(s' \mid s, a_\sigma) \sum_{o \in \mathcal{O}} O(o \mid a_\sigma, s') \alpha_{\sigma(o)}(s')
$$
当智能体处于某一不确定的置信状态 $\mathbf{b}$ 时，执行该条件计划 $\sigma$ 所能获得的数学期望效用，正是该向量与置信分布的**内积点积（点乘）**：
$$
U_\sigma(\mathbf{b}) = \sum_{s \in \mathcal{S}} b(s) \alpha_\sigma(s) = \boldsymbol{\alpha}_\sigma^\top \mathbf{b}
$$
由于内积是关于置信向量 $\mathbf{b}$ 的线性超平面，而在当前时刻智能体显然会在所有候选条件计划集合 $\mathcal{P}_k$ 中选择最优者，故**最优价值函数恒为一组线性超平面的上包络面（Upper Envelope）**：
$$
U_k(\mathbf{b}) = \max_{\sigma \in \mathcal{P}_k} \boldsymbol{\alpha}_\sigma^\top \mathbf{b} = \max_{\boldsymbol{\alpha} \in \Gamma_k} \boldsymbol{\alpha}^\top \mathbf{b}
$$

**Sondik 定理**：有限个线性函数的最大值函数必然是**分段线性且严格凸的（Piecewise Linear and Convex, PWLC）**！
凸性的直觉极为深刻：**处于置信单纯形内部（高不确定性、混沌迷茫）时的价值，永远劣于处于单纯形顶点处（状态完全确定、信息完备）时的价值**。信息在 POMDP 中天生具有正价值！

```julia
# Alpha 向量与条件计划评估实现 (来自官方 Julia 算法实现)
function alphavector_iteration(𝒫::POMDP, M, Γ)
    for k in 1:M.k_max
        Γ = update(𝒫, M, Γ)
    end
    return Γ
end
####################
```

---

## 20.3 线性规划剪枝 (Pruning Dominated Vectors)

尽管理论上存在极其庞大的条件计划集合，但在单纯形几何空间中，**绝大多数 Alpha 向量是完全冗余且被严格支配的（Dominated）**——即没有任何一个合法的置信状态 $\mathbf{b}$ 会使得该向量处于上包络面的顶端。

### 线性规划精确验证支配性
判定某个特定的候选向量 $\boldsymbol{\alpha}^*$ 是否应当被剪枝丢弃，可以通过求解如下线性规划（LP）判定命题：
$$
\begin{aligned}
\max_{\mathbf{b}, \delta} \quad & \delta \\
\text{s.t.} \quad & \boldsymbol{\alpha}^{*\top} \mathbf{b} - \boldsymbol{\alpha}^\top \mathbf{b} \ge \delta \quad (\forall \boldsymbol{\alpha} \in \Gamma \setminus \{\boldsymbol{\alpha}^*\}) \\
& \sum_{s} b(s) = 1, \quad b(s) \ge 0 \quad (\forall s)
\end{aligned}
$$
- 若最优解 $\delta^* \le 0$：表明在置信空间的任何角落，都存在至少一个其他向量的价值大于等于 $\boldsymbol{\alpha}^*$。该向量是一个无用的被支配冗余项，可被**彻底无损剪枝剔除**！
- 若 $\delta^* > 0$：表明在最优解解出的置信点 $\mathbf{b}^*$ 处，该向量具有独特的全局最高效用，必须予以保留。

```julia
# 线性规划精确剪枝算法实现
struct QMDP
    k_max # maximum number of iterations
end

function update(𝒫::POMDP, M::QMDP, Γ)
    𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
    Γ′ = [[R(s,a) + γ*sum(T(s,a,s′)*maximum(α′[j] for α′ in Γ)
        for (j,s′) in enumerate(𝒮)) for s in 𝒮] for a in 𝒜]
    return Γ′
end

function solve(M::QMDP, 𝒫::POMDP)
    Γ = [zeros(length(𝒫.𝒮)) for a in 𝒫.𝒜]
    Γ = alphavector_iteration(𝒫, M, Γ)
    return AlphaVectorPolicy(𝒫, Γ, 𝒫.𝒜)
end
####################
```

---

## 20.4 精确价值迭代 (Exact Value Iteration)

结合向量递归生成与剪枝，经典 POMDP 精确价值迭代（Monahan, 1982）执行以下迭代递推：
1. 从 0 步计划集合 $\Gamma_0 = \{\mathbf{0}\}$ 开始；
2. 在第 $k$ 轮迭代中，对每个动作 $a$ 和每个观测 $o$，分别从上一轮保留的有效集合 $\Gamma_{k-1}$ 中提取子向量进行交叉笛卡尔组合，生成所有的单步前向候选向量；
3. 将所有生成的候选向量汇总，调用线性规划剪枝算法执行完全修剪，输出紧凑的全新有限集合 $\Gamma_k$；
4. 循环直至连续两轮的上包络面最大残差小于收敛容差。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_20_3.png" alt="分段线性凸函数随迭代步数演变" style="max-width: 480px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 20.3：在啼哭婴儿问题中，价值迭代从 1 步计划、2 步计划到 3 步计划的分段线性凸包络面紧致化全景。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_20_4.png" alt="跨步 Alpha 向量复合构造" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 20.4：通过选择初始动作与各个观测子计划构造新的 $k+1$ 步 Alpha 向量的组合逻辑。</p>
  </div>
</div>

::: danger 复杂度极限
尽管剪枝大幅压缩了向量集合，但在最坏情况下，精确价值迭代每步保留的有效 Alpha 向量数量依然随状态和观测数量呈组合爆炸。对于包含 10 个以上物理状态的通用 POMDP，精确价值迭代在工程上往往难以推进至超过 10 步深度，这催生了点基近似算法的发展。
:::

---

## 20.5 本章小结 (Summary)

- **置信空间的马尔可夫转化**：POMDP 在数学上严格等价于定义在连续概率单纯形上的全观测置信 MDP；
- **分段线性凸性（PWLC）**：有限时域最优价值函数恒为一组 Alpha 超平面的上包络极值，凸性揭示了信息与确定性的内在价值；
- **条件计划与向量同构**：每个离散条件树严格同构映射为一个固定的状态期望效用向量 $\boldsymbol{\alpha}$；
- **线性规划剪枝**：通过求解 LP 探测每个超平面是否存在非空的优势领域，能够无损剔除绝大多数被支配的冗余分支；
- **精确解法的理论边界**：精确算法为 POMDP 确立了不可动摇的数学基准，但其极端严苛的双重指数复杂度迫使我们在大规模任务中转向点基近似。

---

## 20.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 20.1 (Exercise 20.1)
**题目**：考虑一个两状态 POMDP，$\mathcal{S} = \{s_1, s_2\}$。当前价值函数由 2 个 Alpha 向量构成：$\boldsymbol{\alpha}_1 = [10, 0]^\top$，$\boldsymbol{\alpha}_2 = [2, 8]^\top$。设置信状态由标量 $b = P(s_1) \in [0, 1]$ 参数化（此时 $P(s_2) = 1 - b$）。求两个超平面相交的临界置信点 $b^*$，并写出分段线性价值函数 $U(b)$ 的解析分段表达式。

**详细解答**：
分别计算两向量在置信点 $b$ 处的点积效用：
- $U_1(b) = b \alpha_1(s_1) + (1 - b) \alpha_1(s_2) = 10 b + 0 (1 - b) = 10 b$；
- $U_2(b) = b \alpha_2(s_1) + (1 - b) \alpha_2(s_2) = 2 b + 8 (1 - b) = 8 - 6 b$；
令两者效用相等求交点：
$$
10 b = 8 - 6 b \implies 16 b = 8 \implies b^* = \frac{8}{16} = \mathbf{0.5}
$$
最优价值函数取两者的上包络最大值：
$$
U(b) = \max(10b, \; 8 - 6b) = \begin{cases} 8 - 6b & (0 \le b \le 0.5) \\ 10b & (0.5 < b \le 1.0) \end{cases}
$$
函数呈现典型的以 $b=0.5$ 为拐点的 V 型连续凸函数，在两端顶点处效用最高（$U(0)=8, U(1)=10$），在中心最模糊处效用最低（$U(0.5)=5$）。

---

### 习题 20.2 (Exercise 20.2)
**题目**：在上题的设定中，若新增了一个候选 Alpha 向量 $\boldsymbol{\alpha}_3 = [4, 4]^\top$。利用几何或代数判断，该向量是否应当被剪枝丢弃？

**详细解答**：
计算 $\boldsymbol{\alpha}_3$ 在任意置信点 $b \in [0, 1]$ 处的效用：
$$
U_3(b) = 4 b + 4 (1 - b) = 4.0 \quad (\text{水平恒定值 4.0})
$$
对比既有包络面：
- 在拐点 $b = 0.5$ 处，既有最小值为 $U(0.5) = 10 \times 0.5 = 5.0$；
- 由于在整个区间 $[0, 1]$ 上恒有 $U(b) \ge 5.0 > 4.0 = U_3(b)$；
- 向量 $\boldsymbol{\alpha}_3$ 在整个置信单纯形上处处被既有超平面完全遮蔽压制，不存在任何一个使得 $\boldsymbol{\alpha}_3^\top \mathbf{b} > U(b)$ 的点。
因此，**向量 $\boldsymbol{\alpha}_3$ 是一个绝对被支配项，必须被完全剪枝彻底剔除**。

---

### 习题 20.3 (Exercise 20.3)
**题目**：在包含 2 个动作、3 个可能观测的 POMDP 中，若第 1 步存在 2 个不同的有效 1 步计划。若不进行任何剪枝，构造所有的 2 步条件计划总共会生成多少个候选树？

**详细解答**：
代入条件计划双重组合递归公式：$|\mathcal{P}_2| = |\mathcal{A}| \cdot |\mathcal{P}_1|^{|\mathcal{O}|}$：
- 动作数 $|\mathcal{A}| = 2$；
- 1 步计划数 $|\mathcal{P}_1| = 2$；
- 观测数 $|\mathcal{O}| = 3$；
计算 2 步计划总数：
$$
|\mathcal{P}_2| = 2 \times 2^3 = 2 \times 8 = \mathbf{16 \text{ 个条件计划}}
$$

---

### 习题 20.4 (Exercise 20.4)
**题目**：在什么情况下，一个原本部分可观测的 POMDP 在最优价值函数上会退化为一条全局平坦的单一水平超平面？

**详细解答**：
当且仅当系统处于**死胡同或各动作无差异的极端对称环境**中（例如所有状态的即时奖励恒为常数 $R(s, a) = c$，或不论采取何种动作均会导致完全相同的随机吸收态）。此时无论处于何种置信分布，最优累积回报恒为常数 $\frac{c}{1 - \gamma}$，集合中仅包含单一 Alpha 向量，其所有状态分量完全相同。

---

### 习题 20.5 (Exercise 20.5)
**题目**：为什么在 POMDP 中，两个不同的条件计划树有可能在数学上映射为同一个完全相同的 Alpha 向量？

**详细解答**：
因为传感器观测可能存在**信息冗余**。例如某个传感器读数 $o_1$ 在物理世界中发生的概率严格为零（$O(o_1 \mid a, s') = 0$）；或者两个不同的动作分支在物理动力学上恰好产生完全对称的期望收益。此时条件计划树即使在这些冗余分支上指定了不同的后续动作，在代数展开求和时对应的权重系数均为零，最终累积积分计算所得的 Alpha 向量分量完全相等。
