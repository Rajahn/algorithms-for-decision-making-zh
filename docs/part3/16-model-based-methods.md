# 第 16 章：基于模型的方法 (Model-Based Methods)

在未知环境的强化学习中，算法通常划分为两大流派：**基于模型的方法（Model-Based Methods）**与**无模型方法（Model-Free Methods）**。

基于模型的方法遵循“**学习环境物理规律 $\to$ 规划动作策略**”的两阶段解耦逻辑：智能体在与未知环境的试错交互中，显式估计或维护环境的转移概率模型 $\hat{T}(s' \mid s, a)$ 与奖励模型 $\hat{R}(s, a)$；随后，智能体利用第 7 章介绍的精确动态规划或第 9 章的在线树搜索算法，在学到的内部虚拟世界模型中求解最优策略。

本章系统探讨基于模型的强化学习理论体系。我们首先介绍基础的**极大似然 MDP 估计**与**确定性等价控制（Certainty Equivalence）**；随后剖析确定性等价因缺乏探索而陷入局部僵局的机制；接着深入推导具备严格多项式样本复杂度理论保证的经典算法——**R-MAX 算法**（面向未知世界的乐观探索）；随后阐述贝叶斯后验采样在马尔可夫决策中的直接推广——**后验采样强化学习（PSRL）**；最后探讨在置信信念空间中推演的**贝叶斯自适应 MDP（Bayes-Adaptive MDP, BAMDP）**。

---

## 16.1 极大似然模型估计 (Maximum Likelihood MDP)

在有限状态有限动作离散 MDP 中，估计转移与奖励模型最直接的方法是极大似然估计。

### 16.1.1 经验统计量统计
智能体维护以下经验计数器：
- $N(s, a)$：在状态 $s$ 下尝试动作 $a$ 的总次数；
- $N(s, a, s')$：在状态 $s$ 下尝试动作 $a$ 且下一时刻成功转移至状态 $s'$ 的次数；
- $\rho(s, a)$：在状态 $s$ 下执行动作 $a$ 所获得的累积即时奖励总和。

转移概率与奖励函数的极大似然闭式估计量为：
$$
\hat{T}(s' \mid s, a) = \frac{N(s, a, s')}{N(s, a)}, \quad \hat{R}(s, a) = \frac{\rho(s, a)}{N(s, a)}
$$
当 $N(s, a) = 0$ 时，模型赋以默认先验（如均匀转移概率 $\hat{T}(s' \mid s, a) = \frac{1}{|\mathcal{S}|}$）。

```julia
# 极大似然 MDP 模型估计实现 (来自官方 Julia 算法实现)
mutable struct MaximumLikelihoodMDP
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    N # transition count N(s,a,s′)
    ρ # reward sum ρ(s, a)
    γ # discount
    U # value function
    planner
end

function lookahead(model::MaximumLikelihoodMDP, s, a)
    𝒮, U, γ = model.𝒮, model.U, model.γ
    n = sum(model.N[s,a,:])
    if n == 0
        return 0.0
    end
    r = model.ρ[s, a] / n
    T(s,a,s′) = model.N[s,a,s′] / n
    return r + γ * sum(T(s,a,s′)*U[s′] for s′ in 𝒮)
end

function backup(model::MaximumLikelihoodMDP, U, s)
    return maximum(lookahead(model, s, a) for a in model.𝒜)
end

function update!(model::MaximumLikelihoodMDP, s, a, r, s′)
    model.N[s,a,s′] += 1
    model.ρ[s,a] += r
    update!(model.planner, model, s, a, r, s′)
    return model
end
####################
```

---

## 16.2 确定性等价控制与探索陷阱 (Certainty Equivalence)

最简单的决策策略是**确定性等价控制（Certainty Equivalence Control）**：在每个时间步，直接将当前基于历史数据经验拟合出的经验模型 $(\hat{T}, \hat{R})$ 当作绝对无误的“真实物理世界模型”，并调用价值迭代或策略迭代求解其最优策略：
$$
\pi_t(s) = \arg\max_a \left[ \hat{R}(s, a) + \gamma \sum_{s'} \hat{T}(s' \mid s, a) U^*(s') \right]
$$

::: danger 致命缺陷：缺乏探索导致的局部闭环锁死
确定性等价控制在理论上**极易陷入严重的次优死锁**。
如果智能体在探索初期偶然发现了一条平庸但稳定的低收益路径，它在经验模型中便会认为这条路径效用很高；对于那些从未涉足过的高价值潜在通道，由于缺乏采样数据，经验模型对其评价可能很低。确定性等价控制器将永远重复走那条已知平庸路径，永远不再去碰触未知状态，导致智能体无法收敛至全局最优策略。
:::

---

## 16.3 R-MAX 算法与 PAC-MDP 保证 (The R-MAX Algorithm)

为了彻底解决确定性等价控制的早熟锁死难题，布拉夫曼与泰嫩霍尔茨（Brafman & Tennenholtz, 2002）提出了具有里程碑意义的 **R-MAX 算法**。

R-MAX 完美将“面对不确定性时的乐观主义”原则具象化为清晰的工程二分法则：

### 16.3.1 已知状态与未知状态二分法
设置一个关键的**经验有效性阈值 $m$**：
- **已知状态（Known State-Action Pair）**：若状态-动作对 $(s, a)$ 的访问次数 $N(s, a) \ge m$，说明已积累了足够的数据，其转移概率与奖励已被高精度测定，直接使用其极大似然估计模型 $(\hat{T}, \hat{R})$；
- **未知状态（Unknown State-Action Pair）**：若访问次数 $N(s, a) < m$，说明认知极度匮乏。R-MAX 采取激进的**极度乐观赋值**：
  - 将该动作的即时奖励直接强制定为**理论最大可能奖励 $R_{\max}$**：
    $$
    R(s, a) = R_{\max}
    $$
  - 将其转移概率强制设为**确定性自环（Self-Loop）**：执行该未知动作将永远停留在当前状态，并在未来无限期地持续享受 $R_{\max}$ 奖励！

```julia
# R-MAX 算法实现
function MDP(model::MaximumLikelihoodMDP)
    N, ρ, 𝒮, 𝒜, γ = model.N, model.ρ, model.𝒮, model.𝒜, model.γ
    T, R = similar(N), similar(ρ)
    for s in 𝒮
        for a in 𝒜
            n = sum(N[s,a,:])
            if n == 0
                T[s,a,:] .= 0.0
                R[s,a] = 0.0
            else
                T[s,a,:] = N[s,a,:] / n
                R[s,a] = ρ[s,a] / n
            end
        end
    end
    return MDP(T, R, γ)
end
####################
```

### 16.3.2 隐式主动探索机制
当调用价值迭代求解由上述二分规则构造的虚拟 MDP 时：
- 未知状态因其虚构的极高天花板收益（$R_{\max} / (1 - \gamma)$），在价值曲面中如同强大的引力源，产生强烈的势能梯度；
- 这一势能梯度自动驱动规划器生成一条**引导智能体从当前已知区域机动前往未知区域的最短路径**！
- 智能体主动前往未知区域执行动作：要么确实发现了高收益通道，要么将未知状态的计数器刷满 $m$ 次，使其转为已知状态并剥离虚拟奖励。

**PAC-MDP 理论保证**：R-MAX 在多项式样本复杂度（Probably Approximately Correct, PAC）意义下，严格保证在绝大多数时间步内输出 $\epsilon$-近最优策略，且所犯错误的样本上限随状态数和动作数仅呈多项式增长。

---

## 16.4 后验采样强化学习 (Posterior Sampling for Reinforcement Learning, PSRL)

R-MAX 的乐观探索机制需要人工设置阈值 $m$ 与最大奖励 $R_{\max}$，若设置保守可能导致过量无效探索。**后验采样强化学习（PSRL, Strens, 2000; Osband et al., 2013）**将汤普森采样的优雅智慧无缝拓展至全状态 MDP。

### 算法机制：
1. 为每个状态-动作对的转移概率赋予狄利克雷先验 $\text{Dir}(\boldsymbol{\alpha}(s, a))$，为奖励赋予正态-逆伽马共轭先验；
2. 在每个幕（Episode）或规划周期开始时：从当前的贝叶斯后验信念中，**随机抽取一个完整的虚拟 MDP 物理模型实例 $\tilde{\mathcal{M}} = (\tilde{T}, \tilde{R})$**；
3. 调用离线动态规划，求解该虚拟 MDP 实例在当前周期下的精确最优策略 $\pi$；
4. 在整幕交互中，智能体**严格遵循该策略 $\pi$ 执行动作**，采集环境转移轨迹；
5. 幕末利用新采集的数据执行贝叶斯后验更新，进入下一轮采样。

```julia
# 后验采样强化学习 (PSRL) 实现
struct FullUpdate end

function update!(planner::FullUpdate, model, s, a, r, s′)
    𝒫 = MDP(model)
    U = solve(𝒫).U
    copy!(model.U, U)
    return planner
end
####################
```

PSRL 通过整幕采样的相干性，自发保证了智能体在时间步上的**深度探索（Deep Exploration）**，彻底避免了单步抖动的浅层试探。

---

## 16.5 贝叶斯自适应 MDP (Bayes-Adaptive MDP, BAMDP)

在纯粹的决策理论视角下，模型不确定性同样可以被转化为一种带有信息状态的特殊 MDP。**贝叶斯自适应 MDP（BAMDP, Duff, 2002）**将环境的物理状态 $s$ 与智能体当前的贝叶斯模型置信参数（充分统计量 $\mathbf{\alpha}$）拼接为一个**超状态（Hyperstate）**：
$$
\mathbf{s}_{\text{hyper}} = (s, \mathbf{\alpha})
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_16_1.png" alt="包含模型不确定性的动态决策网络" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 16.1：包含模型不确定性的动态决策网络展开图。模型参数作为恒定的潜在隐变量在时序上持续影响转移与观测。</p>
</div>

在 BAMDP 中，探索动作的“信息价值”被内生转化为超状态空间转移方程中的确定性项。然而，由于置信计数连续且不可逆增长，BAMDP 的超状态空间是无穷维的，精确求解通常在计算上是不可判定的，工程中通常结合在线蒙特卡洛规划进行局部逼近。

---

## 16.6 本章小结 (Summary)

- **基于模型的解耦优势**：显式建模环境转移与奖励，使算法能够完全在虚拟世界中进行推演规划，样本利用效率显著优于无模型试错；
- **确定性等价的局限**：将经验模型直接当作真理会导致智能体过早停止探索而深陷局部陷阱；
- **R-MAX 的乐观导向**：通过对访问不充分的未知状态人为赋予极大奖励 $R_{\max}$，系统性将探索需求转化为寻找通往未知边界的自动规划轨迹；
- **PSRL 的深度连贯探索**：通过直接从狄利克雷后验抽样全局 MDP 实例并执行全幕策略，兼具近乎理论极限的低遗憾界与极高的计算效率。

---

## 16.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 16.1 (Exercise 16.1)
**题目**：在包含 3 个状态的 MDP 中，某状态 $s_1$ 执行动作 $a_1$ 共被观测到 10 次。后继状态分别出现：$s_1 \to s_1$ 出现 5 次，$s_1 \to s_2$ 出现 3 次，$s_1 \to s_3$ 出现 2 次。若转移概率先验为拉普拉斯均匀先验（$\alpha_k = 1$），求平滑后的转移概率向量 $\hat{T}(\cdot \mid s_1, a_1)$。

**详细解答**：
代入狄利克雷后验期望更新公式：
$$
\hat{T}(s_k \mid s_1, a_1) = \frac{N(s_1, a_1, s_k) + \alpha_k}{\sum_{k'=1}^3 (N(s_1, a_1, s_{k'}) + \alpha_{k'})}
$$
分母为 $(5 + 3 + 2) + (1 + 1 + 1) = 10 + 3 = 13$。
各转移概率分别为：
- $\hat{T}(s_1 \mid s_1, a_1) = \frac{5 + 1}{13} = \mathbf{\frac{6}{13} \approx 0.4615}$；
- $\hat{T}(s_2 \mid s_1, a_1) = \frac{3 + 1}{13} = \mathbf{\frac{4}{13} \approx 0.3077}$；
- $\hat{T}(s_3 \mid s_1, a_1) = \frac{2 + 1}{13} = \mathbf{\frac{3}{13} \approx 0.2308}$。

---

### 习题 16.2 (Exercise 16.2)
**题目**：在 R-MAX 算法中，已知理论最大单步奖励为 $R_{\max} = 10$，折扣因子 $\gamma = 0.9$。求未知状态在自环假设下的无限时域贴现累积效用 $U_{\max}$。

**详细解答**：
在自环假设下，智能体永远停留在该未知状态并获得恒定即时奖励 $R_{\max}$：
$$
U_{\max} = \sum_{t=0}^\infty \gamma^t R_{\max} = \frac{R_{\max}}{1 - \gamma}
$$
代入数值：
$$
U_{\max} = \frac{10}{1 - 0.9} = \frac{10}{0.1} = \mathbf{100.0}
$$

---

### 习题 16.3 (Exercise 16.3)
**题目**：简述 R-MAX 算法中阈值 $m$ 的物理含义。如果将 $m$ 设置得过大或过小，分别会有何后果？

**详细解答**：
- **物理含义**：$m$ 是根据霍夫丁不等式推导出的保证经验转移经验均值以极高概率逼近真实转移物理分布所需的**最小充分统计量样本数**；
- **$m$ 设置过小**（如 $m=1$）：仅访问一两次便轻率认定该状态为已知状态，导致经验模型存在巨大的估计偏差，算法失去继续探索的主动性；
- **$m$ 设置过大**：智能体必须在每一个状态-动作对上耗费巨额的采样成本将其刷满 $m$ 次，导致前期的无效探索时间极度漫长，样本利用率下降。

---

### 习题 16.4 (Exercise 16.4)
**题目**：在后验采样强化学习（PSRL）中，为什么必须在“整幕（Episode）”的尺度上保持采样策略固定，而不是在每个单步（Step）都重新抽取一个新的 MDP 实例？

**详细解答**：
若在每一个单步都重新独立抽取一个新的 MDP，由于每次采样的 MDP 实例不同，各步的最优行动方向会随机互相抵消，智能体将在当前状态周围进行像布朗运动一样的低效随机漫步；
而在整幕尺度上固定同一个 MDP 采样实例，智能体为了实现该特定世界的长程目标，会执行高度自洽、连贯前向的探索动作序列，这种**时间上的相干性（Temporal Consistency）**正是实现高维深度探索的数学核心。

---

### 习题 16.5 (Exercise 16.5)
**题目**：在确定性等价控制中，给出一种通过修改动作选择规则以规避局部次优陷阱的常见工程补丁。

**详细解答**：
最常用的补丁是将贪心动作选择替换为 **$\epsilon$-贪心探索** 或 **玻尔兹曼软最大（Boltzmann / Softmax）探索**：以非零的概率主动尝试非当前最优模型动作；或者在经验奖励中人为叠加一个正比于访问次数倒数的**内在好奇心奖励（Intrinsic Exploration Bonus / 计数奖励 $r_{\text{bonus}} \propto 1 / \sqrt{N(s, a)}$）**。

---

### 习题 16.6 (Exercise 16.6)
**题目**：在贝叶斯自适应 MDP（BAMDP）中，为什么状态转移在物理真实世界中虽然是静态未知的，但在超状态空间中却呈现出“非平稳”的特性？

**详细解答**：
在超状态 $\mathbf{s}_{\text{hyper}} = (s, \mathbf{\alpha})$ 中，置信向量 $\mathbf{\alpha}$ 记录了历史访问的累加计数。由于每次执行动作后计数器单调递增（$\alpha \leftarrow \alpha + 1$），超状态转移网络实际上是一个单向有向无环图，历史状态永远无法回退，这使得相同的物理状态 $s$ 在不同时刻因包含不同的先验置信度而对应完全不同的超状态转移核。

---

### 习题 16.7 (Exercise 16.7)
**题目**：相较于无模型算法，基于模型的强化学习（MBRL）通常在哪些维度具备显著优势？在哪些维度处于劣势？

**详细解答**：
- **优势（Pros）**：
  1. **超高的样本利用率（Sample Efficiency）**：通常比无模型方法快 1 到 2 个数量级，非常适合物理样机交互成本高昂的真实机器人与工业系统；
  2. **任务迁移与目标泛化**：环境动力学规律与具体任务奖励解耦，若改变奖励目标，只需直接在学好的环境模型上重新规划，无需重新采集数据；
- **劣势（Cons）**：
  1. **模型误差累积（Model Exploitation / Bias）**：若动力学极度复杂导致环境模型预测存在微小瑕疵，规划器往往会贪心寻找模型的漏洞欺骗仿真，导致在真实物理环境中性能崩溃。
