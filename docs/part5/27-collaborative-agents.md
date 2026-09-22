# 第 27 章：协同智能体 (Collaborative Agents)

在第 24 至 26 章中，智能体之间普遍存在着潜在的利益冲突或自利对抗。然而，在多机器人编队协同、智能电网区域自主平衡、分布式仓库 AGV 小车调度等一大批核心工业任务中，**所有智能体实际上隶属于同一个团队，拥有完全一致的共同利益目标**！

本章系统探讨多智能体协同决策理论体系。我们首先剖析**分布式部分可观测马尔可夫决策过程（Decentralized POMDP, Dec-POMDP）**的形式化框架与团队协同瓶颈；随后引入基于空间稀疏图因式分解的**网络化分布式模型（ND-POMDP）与协同图（Coordination Graphs）**；最后，作为全书正文的技术高潮，本章推出**全书所有决策模型的全局大一统亲缘谱系图（Taxonomy of Decision Making Models）**，将横跨 27 个章节的所有核心数理模型梳理串联成一幅恢弘壮丽的技术全景画卷。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（足球队无声的默契配合）**：Dec-POMDP 聚焦于全团队追求同一目标的协作系统（没有自私背叛）。但正因为彼此物理隔离且无法随时无线电喊话，每个人只能凭自己的局部眼睛推测队友此刻在干嘛。
- **协同图（Coordination Graphs）的空间减熵**：
  - 城市有一万个红绿灯，但朝阳区的红绿灯不需要管海淀区的红绿灯；
  - 协同图利用空间稀疏性将团队奖励拆解为局部超图因子，在局部邻域之间传递消息，打破了 NEXP-完全的复杂度深渊。
- **全书终极谱系（图 27.2 的大一统视野）**：
  - 单步 $\to$ 时序 MDP；
  - 完全观测 $\to$ 部分可观测 POMDP；
  - 已知模型 $\to$ 强化学习 RL；
  - 单人孤独决策 $\to$ 多人博弈 POSG 与团队协同 Dec-POMDP。
  全书横跨 27 章的所有数理模型，在时间、观测、模型与主体四维时空坐标系中达成了完美的逻辑大一统！
:::
---

## 27.1 分布式 POMDP (Dec-POMDP)

**分布式 POMDP（Dec-POMDP, Bernstein et al., 2002）**用于建模多个智能体在具有部分可观测性且**通信受限**的环境下为了同一个团队目标而奋斗的场景。

其数学七元组与 POSG 几乎完全一致，但具有一个核心决定性特征——**共同团队奖励（Shared Team Reward）**：
$$
R_1(s, \mathbf{a}) = R_2(s, \mathbf{a}) = \dots = R_n(s, \mathbf{a}) = R(s, \mathbf{a})
$$
所有智能体共同追求最大化全团队的累积贴现总收益：
$$
\max_{\pi_1, \dots, \pi_n} \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_1^{(t)}, \dots, a_n^{(t)}) \right]
$$

### NEXP-Complete 复杂度深渊
尽管智能体之间没有任何勾心角斗的自私博弈，但伯恩斯坦等人（Bernstein et al., 2002）严格证明：**有限时域 Dec-POMDP 的精确求解属于 NEXP-完全问题（NEXP-Complete）！**
这意味着其计算难度超越了常规的 NP-hard，甚至比国际象棋博弈树搜索更加艰巨。其根源在于：在缺乏即时中心通信的情况下，每个分布式个体必须在去中心化的局部观测下，维持与全队所有队友的默契步调一致。

```julia
# Dec-POMDP 团队策略表示与评估算法实现 (来自官方 Julia 算法实现)
struct DecPOMDP
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    𝒪  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # reward function
end
####################
```

---

---

## 27.2 Dec-POMDP 的重要子模型分类 (Subclasses of Dec-POMDP)

为了在工程中规避通用的 NEXP-完全复杂度，学术界识别出若干在特定物理假设下退化的高效子模型（见原书表 27.2 汇总）：

| 子模型名称 | 核心特征假设 | 计算复杂度 | 典型应用场景 |
| :--- | :--- | :--- | :--- |
| **MMDP** (Multiagent MDP) | 具有集中式共享传感器，所有智能体**拥有瞬时完全可观测性** | **P**（等价于单智能体全观测 MDP） | 集中式控制塔指挥的无人机机队 |
| **Dec-MDP** (Decentralized MDP) | 联合全观测性：单一智能体仅具有局部观测，但**所有智能体的观测组合起来可以无损恢复真实全局物理状态** | **NEXP-complete**（最坏情况下仍为指数） | 多相机全覆盖监控网络 |
| **ND-POMDP** (Networked Distributed) | 智能体之间的物理交互仅存在于**局部网络超图边（Locally Interacting）**内 | **NP-complete**（随树宽指数增长，显著低于 NEXP） | 城市区域交通信号灯协同控制 |
| **TI-Dec-POMDP** (Transition-Independent) | 状态转移完全独立：自身动作仅影响自身物理状态，仅在**全局奖励函数中存在团队耦合** | **NP-complete** | 分布式巡逻与协同搜救搜索任务 |

---

## 27.3 迭代最优反应与 JESP 算法 (Joint Equilibrium-Based Search, JESP)

求解 Dec-POMDP 最实用的局部近似算法是**联合平衡搜索算法（JESP, Nair et al., 2003）**：
1. 初始阶段：随机初始化所有智能体的局部策略 $(\pi_1^{(0)}, \dots, \pi_n^{(0)})$；
2. 在第 $k$ 轮循环中，固定其余 $n-1$ 个队友的当前策略保持绝对不变；
3. **对于被选中的智能体 $i$**：其余队友的固定策略在数学上被直接吸纳并固化为物理环境转移的一部分，从而将该智能体的决策命题**直接退化为一个标准的单智能体 POMDP**！
4. 调用第 21 章的 PBVI 或第 22 章的在线规划，求解该智能体的最优单边策略 $\pi_i^{(k+1)}$；
5. 轮流遍历所有智能体，循环往复直至所有智能体的策略达成**局部联合纳什均衡**，无法再单边改进。

---

## 27.4 启发式多智能体搜索：MAA* 算法 (Multiagent A* Search)

为了求解有限时域 Dec-POMDP 的全局最优协同策略，**MAA\* 算法（Szer, Charpillet & Zilberstein, 2005）**将搜索空间定义在由“全团队局部策略树组合”构成的元搜索树上：
- 节点：全团队在前 $t$ 步的联合局部决策树前缀；
- 估价函数：$f(\boldsymbol{\pi}_{1:t}) = g(\boldsymbol{\pi}_{1:t}) + h(\boldsymbol{\pi}_{1:t})$；
- 启发式上界 $h$ 通常采用底层完全可观测 MMDP 或松弛 QMDP 的全局最优值；
- 利用经典的 $A^*$ 优先队列机制展开分支，在满足可采纳上界保证下，能够严格证明找到全局最优的去中心化团队协同策略。

---

## 27.5 网络化分布式模型与协同图与协同图 (ND-POMDP & Coordination Graphs)

在很多大规模团队系统（如城市交通信号灯联动）中，并非所有智能体之间都存在物理交集，相互作用往往具有强烈的**局部空间稀疏性**。

**网络化分布式模型（ND-POMDP, Nair et al., 2005）**与**协同图（Coordination Graphs）**通过将全局奖励函数因式分解为局部超图因子的叠加：
$$
R(s, \mathbf{a}) = \sum_{e \in \mathcal{E}} R_e(s_e, \mathbf{a}_e)
$$
式中每个超边 $e$ 仅涉及极少数局域相邻的几个智能体子集。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_27_1.png" alt="ND-POMDP 五智能体超图结构" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 27.1：一个包含 5 个智能体和 3 个局部超边的 ND-POMDP 协同图拓扑结构。因子分解消除了全连接耦合。</p>
</div>

利用协同图，系统可以调用第 3 章介绍的和-积变量消除法，在局部子群之间执行分布式消息传递，从而在多项式时间内求解大规模团队的近似协同行动。

```julia
# 协同图局部消息传递实现
struct DecPOMDPDynamicProgramming
    b   # initial belief
    d   # depth of conditional plans
end

function solve(M::DecPOMDPDynamicProgramming, 𝒫::DecPOMDP)
    ℐ, 𝒮, 𝒜, 𝒪, T, O, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.𝒪, 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    R′(s, a) = [R(s, a) for i in ℐ]
    𝒫′ = POMG(γ, ℐ, 𝒮, 𝒜, 𝒪, T, O, R′)
    M′ = POMGDynamicProgramming(M.b, M.d)
    return solve(M′, 𝒫′)
end
####################
```

---

## 27.6 全书决策制定模型大一统终极谱系大一统终极谱系 (The Grand Taxonomy of Models)

历经全书 27 个章节的系统推导，我们从最简单的单步确定性选择，一步步攀登至不确定性环境下最前沿的多主体交互系统。所有这些看似庞杂的模型，在底层数理逻辑上具有紧密血脉相承的亲缘演化拓扑！

下图 27.2 绘制了**全书所有决策模型的全局大一统亲缘演化图谱**：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_27_2.png" alt="全书决策模型大一统终极谱系图" style="max-width: 720px; width: 100%; display: inline-block; border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 8px;">图 27.2：全书决策制定模型全局终极谱系树。箭头表示模型的继承与一般化泛化方向。</p>
</div>

### 谱系演化四象限法则：
1. **时间维度延展**：简单单步决策（第 6 章）$\xrightarrow{+时间}$ 马尔可夫决策过程 MDP（第 7 章）；
2. **知识认知延展**：已知物理模型 $\xrightarrow{+未知模型}$ 强化学习 RL（第 15 ~ 18 章）与贝叶斯自适应 BAMDP（第 16 章）；
3. **传感器感知延展**：完全可观测 MDP $\xrightarrow{+观测噪声}$ 部分可观测 POMDP（第 19 ~ 23 章）；
4. **决策主体延展**：单智能体 POMDP $\xrightarrow{+多主体对立}$ 随机博弈 POSG（第 24 ~ 26 章）；单智能体 POMDP $\xrightarrow{+多主体协同}$ 分布式系统 Dec-POMDP（第 27 章）。

---

## 27.7 本章小结 (Summary)

- **协同与共赢的崇高范式**：Dec-POMDP 聚焦于全团队利益最大化，构成了多机器人集群协同的标准数学基石；
- **去中心化执行的理论天花板**：信息通信隔离引发的 NEXP-完全复杂度深渊，决定了必须依赖空间超图因子分解（ND-POMDP）；
- **全书终极大一统**：横跨 27 个章节的全部数理模型，在时间、观测、模型与主体四重坐标系下达成完美的结构同构闭环。

---

## 27.8 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 27.1 (Exercise 27.1)
**题目**：在包含 2 个智能体的 Dec-POMDP 中，设双方在做出动作前能够进行**瞬间、无延迟、无限带宽且无噪声的完全双向无线电通信**。此时该系统退化为什么类型的决策模型？

**详细解答**：
当两个智能体能够实时交换各自的所有私有观测时，全团队拥有一个统一的集中式联合观测向量 $\mathbf{o}_t = (o_1^{(t)}, o_2^{(t)})$。
任何去中心化的信息壁垒彻底瓦解，系统退化为一个拥有单一集中式大脑、联合动作空间为 $\mathcal{A}_1 \times \mathcal{A}_2$ 的**标准单智能体 POMDP（Centralized POMDP）**！

---

### 习题 27.2 (Exercise 27.2)
**题目**：在图 27.1 所示的 5 智能体协同图中，若智能体 1 与智能体 5 之间没有直接或间接的超边连边。说明两者在单步最优动作选择上是否满足条件独立性。

**详细解答**：
满足条件独立性。
根据协同图的因子分解定理，全局收益可以表达为各超边独立奖励的和。利用变量消除法消除中间处于隔离边界上的节点后，智能体 1 的动作与智能体 5 的动作在目标函数中完全不存在任何交叉乘积项，两者在单步优化中实现完全局部解耦。

---

### 习题 27.3 (Exercise 27.3)
**题目**：回顾全书大一统谱系图（图 27.2），简要写出从最简的“单步简单决策”逐步演化至“Dec-POMDP”的数学约束逐步泛化路径。

**详细解答**：
1. **简单决策 $\to$ MDP**：引入时序马尔可夫转移算子与状态反馈闭环，由单步静态扩展为无限贴现时域；
2. **MDP $\to$ POMDP**：去除环境状态直接可测假设，引入条件观测分布 $O(o \mid a, s')$，状态被泛化为单纯形置信分布；
3. **POMDP $\to$ Dec-POMDP**：将单一决策者拓展为多个物理隔离、各自拥有局部传感器私有观测的去中心化分布式决策团队，优化目标统一为团队共同奖励。
