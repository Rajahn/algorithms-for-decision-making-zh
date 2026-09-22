# 第 17 章：无模型方法 (Model-Free Methods)

在上一章中，基于模型的方法需要显式学习环境的转移概率矩阵 $T(s' \mid s, a)$。当环境状态空间极大时，不仅估计高阶多维转移矩阵本身耗费巨量显存，而且微小的模型拟合偏差在经过多步规划后会被指数级放大。

**无模型强化学习（Model-Free Reinforcement Learning）**采取了一种更为纯粹、轻量且强大的路径：**智能体完全放弃构建环境物理模型的企图，直接通过与环境的真实试错交互经验，增量学习动作价值函数 $Q(s, a)$ 或状态价值函数 $V(s)$**。

本章系统探讨经典的时序差分（Temporal Difference, TD）无模型强化学习理论。我们首先阐述增量均值估计与罗宾斯-门罗收敛条件；随后深入推导现代强化学习最具代表性的离策略经典算法——**Q-Learning（沃特金斯算法）**；接着对比剖析在策略算法 **SARSA** 及其在风险规避中的行为差异；随后引入贯通单步 TD 与全蒙特卡洛回报的**资格迹（Eligibility Traces）与 $\text{SARSA}(\lambda)$**；最后探讨融入线性特征基函数的参数化无模型逼近。

---

## 17.1 增量估计与时序差分原理 (Incremental Estimation & TD)

设智能体希望估计某个静态或非平稳目标随机变量的数学期望 $\mu = \mathbb{E}[X]$。

给定相继到来的独立观测序列 $x_1, x_2, \dots, x_k$，经验算术均值可展开为递推更新形式：
$$
\hat{\mu}_k = \frac{1}{k} \sum_{i=1}^k x_i = \hat{\mu}_{k-1} + \frac{1}{k} (x_k - \hat{\mu}_{k-1})
$$
将系数 $\frac{1}{k}$ 推广为通用的**学习率步长序列 $\alpha_k \in (0, 1]$**，得到强化学习最核心的通用增量更新骨架：
$$
\text{新估计值} \leftarrow \text{旧估计值} + \alpha_k \cdot \left( \text{目标信号} - \text{旧估计值} \right)
$$
式中括号内项被称为**预测误差（Prediction Error）**。

### 罗宾斯-门罗收敛定理 (Robbins-Monro Conditions, 1951)
为使随机逼近算法在包含噪声的环境中以概率 1 严格收敛至真实条件期望，学习率序列必须满足：
$$
\sum_{k=1}^\infty \alpha_k = \infty \quad (\text{步长总和发散：保证足以克服初始猜测偏差并覆盖任意远的状态}), \quad \sum_{k=1}^\infty \alpha_k^2 < \infty \quad (\text{平方和收敛：保证方差随时间完全衰减})
$$

```julia
# 增量估计抽象数据结构实现 (来自官方 Julia 算法实现)
mutable struct IncrementalEstimate
	μ # mean estimate
	α # learning rate function
	m # number of updates
end

function update!(model::IncrementalEstimate, x)
	model.m += 1
	model.μ += model.α(model.m) * (x - model.μ)
	return model
end
####################
```

---

## 17.2 Q-Learning 算法 (Watkins, 1989)

**Q-Learning** 是由克里斯·沃特金斯（Chris Watkins）在其 1989 年博士论文中提出的里程碑式算法。它是一种典型的**离策略（Off-Policy）时序差分控制算法**。

### 17.2.1 算法机制与更新公式
智能体在当前状态 $s$ 执行动作 $a$（动作可以由某种探索策略如 $\epsilon$-贪心决定），观察到环境反馈的即时奖励 $r$ 与后继状态 $s'$。
Q-Learning 将贝尔曼最优性方程的右端直接作为自举目标：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in \mathcal{A}} Q(s', a') - Q(s, a) \right]
$$
式中 $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ 称为 **TD 误差（Temporal Difference Error）**。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_17_1.png" alt="Q-Learning 在六边形世界中的收敛过程" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 17.1：在六边形世界中，Q-Learning 随着探索轨迹数量递增（从 50 条到 100 条）逐步逼近最优动作价值函数的演化过程。</p>
</div>

### 17.2.2 离策略性质的核心威力
**行为策略（Behavior Policy）与目标策略（Target Policy）的解耦**：
- 智能体在采样交互时可以遵循任意充分探索的行为策略（如完全随机走动或 $\epsilon$-贪心）；
- 但更新目标中的项始终使用 $\max_{a'} Q(s', a')$，这意味着更新目标**始终严格假设后继动作遵循纯正的全局最优贪心策略**！
- 沃特金斯与达扬（Watkins & Dayan, 1992）严格证明：只要所有状态-动作对在无限时域内被持续访问，且学习率满足罗宾斯-门罗条件，表格型 Q-Learning **以概率 1 收敛至全局唯一的真实最优动作价值 $Q^*$**。

```julia
# Q-Learning 核心算法实现
mutable struct QLearning
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
end

lookahead(model::QLearning, s, a) = model.Q[s,a]

function update!(model::QLearning, s, a, r, s′)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + γ*maximum(Q[s′,:]) - Q[s,a])
    return model
end
####################
```

---

## 17.3 SARSA：在策略时序差分控制 (Rummery & Niranjan, 1994)

与 Q-Learning 的激进乐观不同，**SARSA** 是一种**在策略（On-Policy）控制算法**。其命名源于单步经历的五元组时序事件：
$$
(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})
$$

### 17.3.1 更新公式
智能体在后继状态 $s'$ 下，不取假设的最大值 $\max_{a'} Q(s', a')$，而是**直接按照当前探索策略（如 $\epsilon$-贪心）实际采样抽出的下一个真实执行动作 $a'$ 的价值来进行更新**：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_17_2.png" alt="SARSA 在六边形世界中的收敛过程" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 17.2：SARSA 算法在六边形世界中随着交互轮次递增的动作价值函数演化过程。</p>
</div>

### 17.3.2 Q-Learning 与 SARSA 的行为哲学差异（悬崖行走实验）
- **Q-Learning（激进探险家）**：直接学习最优策略的无偏上界，在悬崖行走（Cliff Walking）任务中会选择贴着悬崖边缘的最短路径（因为最优策略假定下一动作绝对不失误）；但在训练探索期，由于 $\epsilon$ 随机扰动，它会频繁失足跌落悬崖；
- **SARSA（谨慎现实主义者）**：学习的是包含自身探索随机性在内的实际执行策略的期望价值。由于感知到自己在边缘有 $\epsilon$ 概率失足，它会自主学习一条绕开悬崖边缘、更远但更安全的迂回路线。

```julia
# SARSA 算法实现
mutable struct Sarsa
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    α # learning rate
    ℓ # most recent experience tuple (s,a,r)
end

lookahead(model::Sarsa, s, a) = model.Q[s,a]

function update!(model::Sarsa, s, a, r, s′)
    if model.ℓ != nothing
        γ, Q, α, ℓ = model.γ, model.Q, model.α,  model.ℓ
        model.Q[ℓ.s,ℓ.a] += α*(ℓ.r + γ*Q[s,a] - Q[ℓ.s,ℓ.a])
    end
    model.ℓ = (s=s, a=a, r=r)
    return model
end
####################
```

---

## 17.4 资格迹与 $\text{SARSA}(\lambda)$ (Eligibility Traces)

单步 TD 算法仅将即时奖励回传给前一个紧邻状态，当奖励极其稀疏时（如走数千步仅在终点获得一次胜利回报），单步价值传播极其缓慢。

**资格迹（Eligibility Traces）**提供了一种短期记忆机制，用标量 $N(s, a)$ 记录每个状态-动作对近期被访问的频次与时间新鲜度：
- **累积迹更新（Accumulating Trace）**：访问时递增 $N(s, a) \leftarrow N(s, a) + 1$；
- **时间衰减**：未被访问的状态按几何衰减 $N(s, a) \leftarrow \gamma \lambda N(s, a)$。

在每个时间步计算出单步标量 TD 误差 $\delta_t$ 后，**同时沿网络所有具备非零资格迹的状态执行并行回传更新**：
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \delta_t N(s, a) \quad (\forall s, a)
$$
参数 $\lambda \in [0, 1]$ 完美将单步更新（$\lambda = 0$）与全轨迹蒙特卡洛（$\lambda = 1$）无缝统一。

---

## 17.5 线性函数逼近 Q-Learning (Linear Function Approximation)

当状态空间连续时，我们用线性基函数拟合每个动作的价值曲面：
$$
\hat{Q}(s, a; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(s, a)
$$
关于参数的**半梯度（Semi-Gradient）更新规则**为：
$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \left[ r + \gamma \max_{a'} \hat{Q}(s', a'; \boldsymbol{\theta}) - \hat{Q}(s, a; \boldsymbol{\theta}) \right] \boldsymbol{\phi}(s, a)
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_17_3.png" alt="线性 Q 学习在山地车中的收敛" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 17.3：利用线性基函数逼近 Q-Learning 求解连续山地车任务所获得的全局效用曲面（左）与三动作边界策略（右）。</p>
</div>

```julia
# 线性函数逼近 Q-Learning 实现
mutable struct SarsaLambda
    𝒮 # state space (assumes 1:nstates)
    𝒜 # action space (assumes 1:nactions)
    γ # discount
    Q # action value function
    N # trace
    α # learning rate
    λ # trace decay rate
    ℓ # most recent experience tuple (s,a,r)
end

lookahead(model::SarsaLambda, s, a) = model.Q[s,a]

function update!(model::SarsaLambda, s, a, r, s′)
    if model.ℓ != nothing
        γ, λ, Q, α, ℓ = model.γ, model.λ, model.Q, model.α, model.ℓ
        model.N[ℓ.s,ℓ.a] += 1
        δ = ℓ.r + γ*Q[s,a] - Q[ℓ.s,ℓ.a]
        for s in model.𝒮
            for a in model.𝒜
                model.Q[s,a] += α*δ*model.N[s,a]
                model.N[s,a] *= γ*λ
            end
        end
    else
    	model.N[:,:] .= 0.0
    end
    model.ℓ = (s=s, a=a, r=r)
    return model
end
####################
```

---

## 17.6 本章小结 (Summary)

- **完全摆脱模型枷锁**：无模型方法无需耗费算力拟合复杂的转移矩阵，直接以环境生成的单步转移样本为驱动；
- **离策略的 Q-Learning**：通过直接将最大动作价值作为更新目标，具备在探索性行为下直奔最优不动点的非凡鲁棒性；
- **在策略的 SARSA**：将探索带来的物理风险内生纳入价值考量，在安全关键控制中展现出高度的现实谨慎性；
- **资格迹的时间加速**：通过维护短时几何衰减记忆，使稀疏远端奖励能够在单步内全链路反向泛化回传。

---

## 17.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 17.1 (Exercise 17.1)
**题目**：在增量均值估计中，设当前对某随机变量期望值的估计为 $\hat{\mu} = 10.0$。新到来的采样值为 $x = 16.0$。若学习率设置为固定常数 $\alpha = 0.25$，计算更新后的新估计值。

**详细解答**：
代入增量更新公式：
$$
\hat{\mu}_{\text{new}} = \hat{\mu} + \alpha (x - \hat{\mu}) = 10.0 + 0.25 \times (16.0 - 10.0) = 10.0 + 0.25 \times 6.0 = 10.0 + 1.5 = \mathbf{11.5}
$$

---

### 习题 17.2 (Exercise 17.2)
**题目**：在表格型 Q-Learning 中，当前状态为 $s$，执行动作 $a$ 获得了即时奖励 $r = 2$ 并转移到新状态 $s'$。已知当前 $Q(s, a) = 5.0$。在新状态 $s'$ 下，有两个可选动作 $a_1', a_2'$，其当前的 $Q$ 值分别为 $Q(s', a_1') = 4.0$，$Q(s', a_2') = 6.0$。设折扣因子 $\gamma = 0.9$，学习率 $\alpha = 0.5$。计算此步更新后的 $Q(s, a)$。

**详细解答**：
1. 提取后继状态的最大动作价值：$\max_{a'} Q(s', a') = \max(4.0, 6.0) = 6.0$；
2. 计算时序差分目标（TD Target）：
   $$
   \text{Target} = r + \gamma \max_{a'} Q(s', a') = 2 + 0.9 \times 6.0 = 2 + 5.4 = 7.4
   $$
3. 计算 TD 误差：
   $$
   \delta = \text{Target} - Q(s, a) = 7.4 - 5.0 = 2.4
   $$
4. 执行更新：
   $$
   Q_{\text{new}}(s, a) = 5.0 + 0.5 \times 2.4 = 5.0 + 1.2 = \mathbf{6.2}
   $$

---

### 习题 17.3 (Exercise 17.3)
**题目**：在上题的相同情境下，若使用 SARSA 算法，且智能体根据其 $\epsilon$-贪心探索策略在新状态 $s'$ 下实际抽中并决定执行的是动作 $a_1'$（其 $Q$ 值为 4.0）。计算此时 SARSA 更新后的 $Q(s, a)$。

**详细解答**：
1. SARSA 直接使用实际选中的下一个动作 $a_1'$ 的值：$Q(s', a_1') = 4.0$；
2. 计算 SARSA 的 TD 目标：
   $$
   \text{Target} = r + \gamma Q(s', a_1') = 2 + 0.9 \times 4.0 = 2 + 3.6 = 5.6
   $$
3. 计算 TD 误差：
   $$
   \delta = 5.6 - 5.0 = 0.6
   $$
4. 执行更新：
   $$
   Q_{\text{new}}(s, a) = 5.0 + 0.5 \times 0.6 = 5.0 + 0.3 = \mathbf{5.3}
   $$
（对比可见：Q-Learning 的更新幅度（6.2）显著比在策略的 SARSA（5.3）更为激进乐观）。

---

### 习题 17.4 (Exercise 17.4)
**题目**：在什么条件下，Q-Learning 与 SARSA 的单步更新数学公式完全等价？

**详细解答**：
当且仅当智能体在新状态 $s'$ 下遵循的探索策略**选中的动作恰好正是当前估计具有最大 $Q$ 值的贪心动作**（即 $a' = \arg\max_{a''} Q(s', a'')$）时，有 $Q(s', a') = \max_{a''} Q(s', a'')$，两者的更新公式在数值上完全同一。

---

### 习题 17.5 (Exercise 17.5)
**题目**：为什么在非平稳环境（即环境的真实奖励或转移规律随时间不断漂移）中，学习率通常设置为固定的小常数（如 $\alpha = 0.1$），而不是满足罗宾斯-门罗递减条件的 $\alpha_t = 1/t$？

**详细解答**：
- 若学习率衰减为 $\alpha_t = 1/t$，随着步数增加，后期的学习率将趋近于零，算法会彻底丧失可塑性（Plasticity），对外界物理规律的漂移失去响应能力；
- 保持固定常数步长 $\alpha$ 使得更新目标赋予各个历史样本呈**指数衰减权重（Exponentially Decaying Recency-Weighted Average）**，越接近当前时刻的最新数据权重越高，保证了系统对非平稳环境的常态化自适应跟踪。

---

### 习题 17.6 (Exercise 17.6)
**题目**：简述资格迹中“累积迹”（Accumulating Trace）与“替换迹”（Replacing Trace）在处理频繁重复访问同一状态时的核心机制差异。

**详细解答**：
- **累积迹**：每次访问状态时无条件递增 $N(s, a) \leftarrow N(s, a) + 1$。如果某个自环状态在短时间内被频繁重复访问，其资格迹会迅速累加至极大的数值，导致随后到来的 TD 误差被严重过度放大，引发数值不稳定；
- **替换迹**：每次访问状态时直接将其重置固定为上限值 $N(s, a) \leftarrow 1$。无论此前该状态被访问多少次，资格迹都被牢牢钳位在 1，显著增强了高维函数逼近下的数值稳健性。

---

### 习题 17.7 (Exercise 17.7)
**题目**：在连续山地车任务中，状态包含小车位置 $x \in [-1.2, 0.6]$ 与速度 $v \in [-0.07, 0.07]$。若直接使用线性 Q-Learning，为什么不能仅用原状态维度向量 $[1, x, v]^\top$，而必须使用瓦片编码（Tile Coding）或多项式/高斯基展开？

**详细解答**：
因为最优价值曲面 $Q^*(s, a)$ 在相空间中具有极其强烈的非凸与非线性特征（小车必须先反向冲上左侧山坡蓄积势能才能越过右侧山顶）。仅用原状态 $[1, x, v]^\top$ 只能拟合一阶倾斜平面超曲面，根本无法表达能量积累与反向机动的鞍点几何。必须通过非线性基展开（升维映射）才能将非线性流形在特征空间中转化为线性可分平面。
