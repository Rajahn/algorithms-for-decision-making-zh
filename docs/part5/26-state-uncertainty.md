# 第 26 章：状态不确定性博弈 (State Uncertainty)

在第 25 章探讨随机博弈时，我们假设所有智能体在每一个时刻均能完全、清晰地观测到底层物理世界的真实全局状态 $s$。

然而，当多智能体交互置于不完全信息的真实物理世界时，面临着复合的终极不确定性：**每个智能体不仅无法完全看清真实物理世界状态，而且无法完全看透其他对手的私有信息与观测历史**。例如在德州扑克中，玩家不仅不知道牌堆剩余底牌，更不知道对手底牌；在潜艇对抗博弈中，双方声呐均受海洋杂波遮挡。

本章系统探讨多智能体系统在状态不确定性下的最通用理论模型——**部分可观测随机博弈（Partially Observable Stochastic Games, POSG）**。我们首先形式化 POSG 的数学体系；随后深入剖析极具挑战性的**交互式置信状态（Interactive Belief States）**与“我认为你认为我认为……”的无限递归信念难题；最后阐述多智能体条件计划与 Alpha 向量超平面表示。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（终极不确定性：迷雾中的扑克牌局）**：POSG 是决策理论的珠穆朗玛峰——你既看不清客观世界的真实物理状态（盲人摸象），又看不清对手手里拿了什么底牌（各怀鬼胎）。
- **心智镜像的无限套娃难题**：
  - 我想预测对手下一步出什么牌，必须推断“他当前的置信是什么”；
  - 而他的置信，又取决于“他认为我认为他的置信是什么”；
  - I-POMDP 通过设定有限认知深度截断（例如在 Level 2 停止），将对手的信念概率化融入扩展状态空间，建立了多智能体对抗中“知己知彼”的严密数学滤网。
:::
---

## 26.1 部分可观测随机博弈形式化 (POSG Formulation)

**部分可观测随机博弈（POSG）**是强化学习与博弈论迄今为止最完备、最具普适性的统一形式化模型。

一个 POSG 由元组 $(\mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, \mathcal{R}, \gamma)$ 精确定义：
1. **智能体集合 $\mathcal{I} = \{1, \dots, n\}$**；
2. **隐藏物理状态空间 $\mathcal{S}$**；
3. **联合动作空间 $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_n$**；
4. **联合观测空间 $\mathcal{O} = \mathcal{O}_1 \times \dots \times \mathcal{O}_n$**：每个智能体拥有自身独立的私有传感器通道，接收私有观测 $o_i \in \mathcal{O}_i$；
5. **状态转移模型 $T(s' \mid s, \mathbf{a})$**；
6. **观测模型 $O(\mathbf{o} \mid \mathbf{a}, s')$**：在执行联合动作并发生物理转移后，为所有智能体联合生成私有观测向量 $\mathbf{o} = (o_1, \dots, o_n)$ 的条件概率分布；
7. **独立奖励函数集合 $\mathcal{R} = \{R_1, \dots, R_n\}$**，其中 $R_i(s, \mathbf{a})$ 为智能体 $i$ 的即时收益。

```julia
# POSG 核心数据结构实现 (来自官方 Julia 算法实现)
struct POMG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    𝒪  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # joint reward function
end
####################
```

---

---

## 26.2 POSG 策略评估与马尔可夫链展开 (Policy Evaluation in POSG)

当所有智能体的策略剖面 $\boldsymbol{\pi} = (\pi_1, \dots, \pi_n)$ 形式化为确定性条件计划树或有限状态控制器时，我们可以在联合状态-历史空间中对其进行精确的全局策略评估。

定义系统在时刻 $t$ 的**全局状态剖面**为底层真实物理状态与所有智能体各自当前观测历史序列的元组：
$$
\mathbf{s}_{\text{global}} = (s, h_1, \dots, h_n)
$$
全系统的转移规律严格构成一个标准马尔可夫链。对任意智能体 $i$，其在初始物理分布与初始信念下的累积贴现总效用 $U_i(\boldsymbol{\pi})$，可通过建立联合状态转移概率矩阵并通过高斯消元求逆直接解析求解。

---

## 26.3 POSG 中的纳什均衡与序列均衡 (Nash Equilibrium in POSG)

在部分可观测博弈中，纳什均衡的概念拓展为**贝叶斯纳什均衡（Bayesian Nash Equilibrium）**与**序列均衡（Sequential Equilibrium, Kreps & Wilson, 1982）**：
- 策略剖面 $\boldsymbol{\pi}^* = (\pi_1^*, \dots, \pi_n^*)$ 构成纳什均衡，当且仅当对于任意智能体 $i$，在给定对手策略 $\boldsymbol{\pi}_{-i}^*$ 的条件下，智能体 $i$ 无法通过单方面修改自身从私有观测历史到动作的任何一个局部映射来获取更高期望回报：
  $$
  U_i(\pi_i^*, \boldsymbol{\pi}_{-i}^*) \ge U_i(\pi_i, \boldsymbol{\pi}_{-i}^*) \quad (\forall \pi_i)
  $$
- 序列均衡进一步要求智能体在每一个信息集内部，其信念评估必须与博弈全局的策略执行一致，排除了不可信的离线信念假设。

---

## 26.4 POSG 动态规划与多智能体 Alpha 剪枝 (Dynamic Programming for POSG)

类似于单智能体 POMDP，有限时域 POSG 也可以通过自底向上的动态规划进行逆推：
1. 每个智能体的条件计划树对应一个超平面 Alpha 向量；
2. 在第 $k$ 轮规划中，智能体 $i$ 必须针对对手可能采取的全部 $k-1$ 步条件计划组合，生成候选响应向量；
3. **多主体劣势策略剪枝（Multiagent Pruning）**：通过求解双重线性规划，剔除那些无论对手采用何种混合策略、且在任何物理置信点下均被支配的劣质条件计划；
4. 随着时域增长，多智能体交叉组合数量极其庞大，促使实际算法广泛采用基于启发式采样的近似剪枝。

---

## 26.5 交互式置信状态与无限递归心智 (Interactive Belief States)

在单智能体 POMDP 中，置信状态是定义在物理状态上的单一概率分布 $b(s) \in \Delta^{|\mathcal{S}|-1}$。但在 POSG 中，事情变得极其诡异：
- 智能体 $i$ 想要预测对手 $j$ 下一步会采取什么动作，必须推测对手当前的置信状态 $b_j$ 是什么；
- 而对手 $j$ 的置信状态又取决于对手对智能体 $i$ 的置信评估；
- 从而引发哲学上著名的**无限递归心智镜像（Infinite Regress of Beliefs）**：“我认为（Level 1）你认为（Level 2）我认为（Level 3）……”

### 交互式 POMDP (I-POMDP, Gmytrasiewicz & Doshi, 2005)
为了在有限计算资源内截断该无限嵌套，I-POMDP 引入了有界认知层级：
$$
b_i^{(k)} \in \Delta(\mathcal{S} \times \Theta_{-i}^{k-1})
$$
智能体维持一个联合置信分布，同时对**物理状态 $s$** 以及**对手所处的可能心智模型参数 $\theta_{-i}$** 进行贝叶斯联合后验更新。

```julia
# 交互式状态不确定性更新实现
function lookahead(𝒫::POMG, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    u′ = sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return R(s,a) + γ*u′
end

function evaluate_plan(𝒫::POMG, π, s)
    a = Tuple(πi() for πi in π)
    U(o,s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π,o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫, U, s, a)
end

function utility(𝒫::POMG, b, π)
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u))
end
####################
```

---

## 26.6 多智能体条件计划与 Alpha 向量鞍点曲面与 Alpha 向量

类似于单智能体 POMDP，在 POSG 中每个智能体的有限时域策略同样可以表示为多智能体条件计划树。
由于系统存在多个参与者，智能体 $i$ 的一个条件计划 $\sigma_i$ 在面对对手策略剖面 $\boldsymbol{\sigma}_{-i}$ 时，在置信单纯形上同样诱导出线性的期望收益超平面：
$$
U_i(\mathbf{b}) = \max_{\sigma_i} \min_{\boldsymbol{\sigma}_{-i}} \boldsymbol{\alpha}_{\sigma_i, \boldsymbol{\sigma}_{-i}}^\top \mathbf{b}
$$
在零和状态不确定性博弈中，全局价值函数表现为分段线性凸函数（PWLC）与凹函数上确界的极其复杂的复合鞍点曲面。

---

## 26.7 本章小结 (Summary)

- **终极不确定性大一统**：POSG 完美融合了状态不完全可观测性与多智能体策略对抗互动；
- **无限递归心智的解构**：交互式置信状态通过将对手的信念内生编码为扩展状态空间的概率测度，形式化破译了“知己知彼”的数学本质；
- **计算复杂度的珠穆朗玛峰**：通用有限时域 POSG 的精确求解已被严格证明属于 NEXP 甚至完全不可判定，必须依赖前向蒙特卡洛采样与局部有限层级截断。

---

## 26.8 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 26.1 (Exercise 26.1)
**题目**：在两智能体 POSG 中，假设物理状态集仅包含 2 个状态，智能体 1 具有 2 个私有观测，智能体 2 具有 3 个私有观测。在一次交互后，环境生成的可能联合观测剖面 $\mathbf{o} = (o_1, o_2)$ 共有多少种组合？

**详细解答**：
联合观测空间为两个私有观测集合的笛卡尔积：$\mathcal{O} = \mathcal{O}_1 \times \mathcal{O}_2$。
组合总数为 $|\mathcal{O}_1| \times |\mathcal{O}_2| = 2 \times 3 = \mathbf{6 \text{ 种联合观测}}$。

---

### 习题 26.2 (Exercise 26.2)
**题目**：简述为什么在单智能体 POMDP 中置信状态是充分统计量，而在多智能体 POSG 中，仅凭智能体自身的单边置信 $b_i(s)$ 无法构成最优决策的充分统计量？

**详细解答**：
在单智能体 POMDP 中，环境动力学仅取决于物理状态与自身动作；但在 POSG 中，后继状态与即时收益强烈依赖于对手的动作 $a_{-i}$，而对手的动作是由对手自身的私有历史观测所驱动的。仅仅知道物理状态分布 $b_i(s)$，完全无法预测对手会在何时基于其私有情报出招，因此智能体必须同时建模对手的信念认知状态。
