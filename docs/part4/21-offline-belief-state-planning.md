# 第 21 章：离线置信状态规划 (Offline Belief State Planning)

正如上一章所揭示的，精确求解 POMDP 伴随着超指数级别的双重维度爆炸，使得精确价值迭代在超过十余个状态的现实问题中难以为继。

为了克服这一计算瓶颈，学术界发展出了极为丰富的**离线近似置信规划（Offline Approximate Planning）**算法体系。本章系统探讨三大主流离线近似技术：
1. **快速上下界启发式算法**：包括假设未来完全可观测的 **QMDP**、紧致压缩的**快速知情界（Fast Informed Bound, FIB）**以及**盲目策略下界（Blind Lower Bound）**；
2. **点基价值迭代（Point-Based Value Iteration, PBVI）**：放弃在全单纯形空间求解，转而在一个有限的代表性置信点集 $B$ 上执行局部超平面备份；
3. **基于界限收缩的启发式引导搜索（SARSOP）**：结合锯齿形上界（Sawtooth Bound）与下界，通过前向可达性分析将算力精准投放于真实最优轨迹可能访问的置信空间核心子流形。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（把有限的算力用在刀刃上）**：上一章的精确算法虽然美妙，但在五六个状态之后就会发生超指数爆炸。PBVI 提出了革命性洞察：**智能体一辈子真正能到达的置信点，只是高维单纯形表面一条微小的低维细线！**
- **工程三叉戟**：
  1. **QMDP（假装下一秒能开天眼）**：直接拿底层 MDP 的最优价值做期望加权，算得飞快，但由于它盲信未来完全可观测，导致它永远不会主动开灯或听诊（丧失主动探索能力）；
  2. **点基价值迭代（PBVI）**：先采样几百个代表性的置信点 $B$，在每个点上只计算并维护一个最贴近的 Alpha 切线超平面，把向量总数死死锁在 $|B|$ 个以内；
  3. **SARSOP（上下界双向夹逼）**：一边维护分段线性下界，一边维护锯齿形上界，只沿着真实最优路径向前展开置信树，剪掉不可达分支，成为现代离线求解的标准工业标杆。
:::
---

## 21.1 快速启发式基线与界限 (Fast Heuristic Bounds)

### 21.1.1 QMDP 启发式算法 (Littman, Cassandra & Kaelbling, 1995)
求解 POMDP 最快的方法是假设环境在当前时间步之后**突然变为完全可观测**：
- 首先调用第 7 章的精确价值迭代，将 POMDP 当作标准 MDP 求解，得到完全可观测下的最优动作价值矩阵 $Q_{\text{MDP}}^*(s, a)$；
- 在实际执行时，面对当前置信状态 $\mathbf{b}$，直接对底层 MDP 的价值进行期望加权：
  $$
  Q_{\text{QMDP}}(\mathbf{b}, a) = \sum_{s \in \mathcal{S}} b(s) Q_{\text{MDP}}^*(s, a)
  $$
- 策略贪心选择：$\pi_{\text{QMDP}}(\mathbf{b}) = \arg\max_a Q_{\text{QMDP}}(\mathbf{b}, a)$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_21_3.png" alt="盲目下界与 BAWS 下界对比" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.1：在啼哭婴儿问题中，盲目策略下界（随着迭代轮数递增）与 BAWS 下界在单纯形空间中的紧致度对比。</p>
</div>

::: info 优缺点权衡
- **优点**：无需任何复杂的超平面备份，仅需秒级时间求解底层 MDP，在“目标导向”明确的任务中表现极佳；
- **致命盲区**：由于 QMDP 盲目假设“未来下一秒就会恢复完全视力”，导致它**完全丧失了主动执行信息收集动作（如开灯、听诊、雷达扫描）的动力**！如果某个任务必须依赖主动探索以消除不确定性，QMDP 会彻底瘫痪。
:::

### 21.1.2 快速知情界 (Fast Informed Bound, FIB)
FIB 通过在价值更新时部分保留观测信息的依赖性，构成了比 QMDP 更紧致的分段线性凸上界：
$$
\alpha_{a, o}(s) = R(s, a) + \gamma \sum_{s'} T(s' \mid s, a) O(o \mid a, s') \alpha_o(s')
$$
FIB 同样可以在多项式时间内快速收敛，且其上界紧致度显著优于 QMDP。

---

## 21.2 点基价值迭代 (Point-Based Value Iteration, PBVI)

皮诺等人（Joelle Pineau et al., 2003）提出了现代 POMDP 离线求解最重要的突破范式——**点基价值迭代（PBVI）**。

### 核心观察：
在真实任务中，智能体能够真正访问到的置信状态，仅仅是整个高维单纯形空间中一条**极其微小的低维可达子流形（Reachable Submanifold）**。为整个单纯形每一个角落去求解超平面的做法是巨大的算力浪费！

### 算法流程：
1. **置信点集收集**：通过在环境中执行随机或探索性策略，采集一组具有代表性的离散置信点集 $\mathcal{B} = \{\mathbf{b}_1, \dots, \mathbf{b}_m\} \subset \Delta^{|\mathcal{S}|-1}$；
2. **点基单步备份（Point-Based Backup）**：在每一轮迭代中，对点集 $\mathcal{B}$ 中的每一个具体置信点 $\mathbf{b}$，**只计算并保留在该点处使其点积达到最大的唯一最优 Alpha 向量**：
   $$
   \boldsymbol{\alpha}_{\mathbf{b}} = \text{Backup}(\mathbf{b}, \Gamma)
   $$
3. 将集合大小严格限制为 $|\Gamma'| \le |\mathcal{B}|$！彻底根除了超指数爆炸！

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_21_4.png" alt="PBVI 近似价值函数演进" style="max-width: 480px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.2：点基价值迭代（PBVI）在啼哭婴儿问题中随着点集迭代的近似价值函数演化。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_21_5.png" alt="随机点更新演进" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.3：随机采样置信点更新的收敛过程。</p>
  </div>
</div>

```julia
# 点基价值迭代 (PBVI) 核心备份实现 (来自官方 Julia 算法实现)
struct FastInformedBound
    k_max # maximum number of iterations
end

function update(𝒫::POMDP, M::FastInformedBound, Γ)
    𝒮, 𝒜, 𝒪, R, T, O, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.𝒪, 𝒫.R, 𝒫.T, 𝒫.O, 𝒫.γ
    Γ′ = [[R(s, a) + γ*sum(maximum(sum(O(a,s′,o)*T(s,a,s′)*α′[j]
        for (j,s′) in enumerate(𝒮)) for α′ in Γ) for o in 𝒪)
        for s in 𝒮] for a in 𝒜]
    return Γ′
end

function solve(M::FastInformedBound, 𝒫::POMDP)
    Γ = [zeros(length(𝒫.𝒮)) for a in 𝒫.𝒜]
    Γ = alphavector_iteration(𝒫, M, Γ)
    return AlphaVectorPolicy(𝒫, Γ, 𝒫.𝒜)
end
####################
```

---

## 21.3 锯齿形上界与 SARSOP 算法

### 21.3.1 锯齿形上界 (Sawtooth Upper Bound)
在单纯形空间的各极值顶点处，真实物理状态完全确定，其价值可由底层 MDP 精确给出。结合离散支撑点集的凸包线性插值，可以构造一个如同锯齿般紧扣真实价值曲面的**严格凸上界（Sawtooth Bound）**：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_21_6.png" alt="锯齿形上界示意" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.4：在啼哭婴儿问题中利用单纯形顶点与内部采样点构建的锯齿形上界（Sawtooth Upper Bound）。</p>
</div>

### 21.3.2 SARSOP：最优可达路径上的极限收缩 (Kurniawati et al., 2008)
**SARSOP（Successively Approximating State-Action-Reward-Observation-Policies）**是公认求解离线 POMDP 性能最强悍的算法之一：
1. 同时维护置信价值的**分段线性凸下界 $\underline{U}$** 与**凸上界 $\bar{U}$**；
2. 仅在初始信念 $b_0$ 出发的可达树前沿展开置信点（Exploratory Belief Expansion）；
3. 利用上下界差距 $\bar{U}(b) - \underline{U}(b)$ 识别最需要细化的局部区域，动态剪枝那些由于确定性次优而不可能被最优策略访问到的置信子树。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_21_7.png" alt="三状态机器维护任务探索性置信展开" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.5：三状态机器替换问题中，探索性置信展开在单纯形表面生成的高密度可达骨架。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_21_8.png" alt="SARSOP 上下界收敛过程" style="max-width: 480px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.6：SARSOP 算法中上界（红色）与下界（蓝色）随迭代推进飞速夹逼收敛至真实曲面。</p>
  </div>
</div>

---

## 21.4 规则网格单纯形剖分 (Freudenthal Triangulation)

对于结构化的连续置信单纯形，另一种系统性近似方法是在单纯形表面建立规整的正交离散网格，并利用**弗洛伊登塔尔三角剖分（Freudenthal Triangulation）**进行局部网格插值与值迭代。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_21_9.png" alt="单纯形网格三角剖分" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.7：二维置信单纯形上的 Freudenthal 规则三角剖分网格。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_21_10.png" alt="机器维护任务网格策略曲面" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.8：在机器维护任务中利用网格插值求解出的全局价值曲面与动作决策边界。</p>
  </div>
</div>

下图 21.9 展示了在置信点备份过程中，一个单步备份如何自发生成一个全新的支撑 Alpha 向量并抬升下界包络面：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_21_11.png" alt="单点置信备份生成 Alpha 向量" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 21.9：在特定置信点 $b$ 处执行备份（Backup）生成切线 Alpha 向量并支撑价值下界的几何图景。</p>
</div>

```julia
# 点基采样与边界更新算法实现
function baws_lowerbound(𝒫::POMDP)
    𝒮, 𝒜, R, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ
    r = maximum(minimum(R(s, a) for s in 𝒮) for a in 𝒜) / (1-γ)
    α = fill(r, length(𝒮))
    return α
end
####################
```

---

## 21.5 本章小结 (Summary)

- **离线近似的必由之路**：全空间连续单纯形的精确求解不可行，算力必须聚焦于实际可达置信子流形；
- **QMDP 的双刃剑**：以忽略未来不确定性为代价换取极速求解，适用于目标驱动型任务，但在强信息依赖场景下失效；
- **点基价值迭代（PBVI）的核心突破**：通过固定离散采样点集 $B$，使每轮生成的 Alpha 向量数量严格受控于 $|B|$，成功将 POMDP 的求解能力推向数百状态大模型；
- **SARSOP 的夹逼哲学**：通过同时收缩上下界并动态剪枝不可达分支，成为现代离线 POMDP 求解的 SOTA 工业级利器。

---

## 21.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 21.1 (Exercise 21.1)
**题目**：在包含 2 个状态的 POMDP 中，底层完全可观测 MDP 在两状态下的最优价值分别为 $Q_{\text{MDP}}^*(s_1, a_1) = 10, Q_{\text{MDP}}^*(s_2, a_1) = 2$ 以及 $Q_{\text{MDP}}^*(s_1, a_2) = 4, Q_{\text{MDP}}^*(s_2, a_2) = 8$。当前置信为 $b(s_1) = 0.6$（$b(s_2) = 0.4$）。利用 QMDP 启发式算法计算各动作的估值，并给出推荐动作。

**详细解答**：
代入 QMDP 期望加权公式：
1. **动作 $a_1$**：
   $$
   Q_{\text{QMDP}}(\mathbf{b}, a_1) = 0.6 \times 10 + 0.4 \times 2 = 6.0 + 0.8 = \mathbf{6.8}
   $$
2. **动作 $a_2$**：
   $$
   Q_{\text{QMDP}}(\mathbf{b}, a_2) = 0.6 \times 4 + 0.4 \times 8 = 2.4 + 3.2 = \mathbf{5.6}
   $$
由于 $6.8 > 5.6$，QMDP 将推荐执行动作 **$a_1$**。

---

### 习题 21.2 (Exercise 21.2)
**题目**：简述为什么 QMDP 永远不可能选择一个“纯信息收集动作”（例如雷达主动开机扫描，该动作即时奖励为负，但不改变物理状态，仅能提供高精度观测）。

**详细解答**：
因为在底层完全可观测 MDP 中，系统假设在执行下一步之后智能体立刻能直接看清真实物理状态。在完全透视的状态下，雷达扫描这种纯开销的信息探测动作是毫无价值的（即在底层 MDP 中其价值必然严格低于直接朝向目标前进的物理动作）。因此其 $Q_{\text{MDP}}^*(s, a)$ 必然处于劣势，加权后 QMDP 绝不可能选择该动作。

---

### 习题 21.3 (Exercise 21.3)
**题目**：在 PBVI 中，若点集包含 $m = 100$ 个采样置信点，动作空间包含 $|\mathcal{A}| = 4$ 个动作，上一轮迭代保留了 100 个 Alpha 向量。在单步更新中，PBVI 最多会输出多少个新的 Alpha 向量？

**详细解答**：
在 PBVI 中，算法对点集 $\mathcal{B}$ 中的每一个置信点 $\mathbf{b}$ 独立求解其最优切线向量，每个置信点仅生成并贡献至多 1 个最优向量。
因此，无论生成过程中间经过多少动作与观测分支组合，**最终保留的新 Alpha 向量数量绝不会超过置信点集的基数，即至多为 $|\mathcal{B}| = \mathbf{100 \text{ 个}}$**！

---

### 习题 21.4 (Exercise 21.4)
**题目**：在 SARSOP 中，为什么在可达置信树中同时维护一个上界 $\bar{U}(b)$ 和一个下界 $\underline{U}(b)$ 能够有效指导启发式采样？

**详细解答**：
差值间隙 $\text{Gap}(b) = \bar{U}(b) - \underline{U}(b)$ 正好量化了当前智能体对该置信状态真实价值的**不确定性宽度**。在从初始信念 $b_0$ 向下选择前向扩展分支时，算法优先顺着使局部间隙最大化的动作-观测分支深入，这保证了每一滴算力都精准投射在那些对最终策略抉择具有最高消除歧义价值的核心瓶颈节点上。
