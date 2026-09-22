# 第 9 章：在线规划 (Online Planning)

在前两章中，无论是离散表格型的精确动态规划，还是基于函数逼近的拟合价值迭代，它们都属于**离线规划（Offline Planning）**范式：在系统实际部署之前，试图为状态空间中的**每一个可能状态**都预先计算出一个全局最优行动。然而，当问题规模庞大或模型极其复杂时，离线计算往往受制于算力瓶颈。

**在线规划（Online Planning）**代表了一种极具威力的互补范式：智能体完全放弃求解全局所有状态策略的企图，转而在每个决策时间步，从**当前实际所处的单一物理状态出发**，在受限的实时计算窗口内进行前瞻性的局部树搜索，推演最优即时动作并立即执行。本章系统探讨在线规划的核心算法。我们首先形式化**滚动时域（Receding Horizon）**机制；随后介绍确定性与随机环境下的**前瞻树搜索（Forward Search）**与**稀疏采样（Sparse Sampling）**；接着深入剖析在现代人工智能中取得里程碑突破的**蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）**及其 UCT 探索准则；随后讨论结合启发式边界的**启发式搜索与实时动态规划（RTDP）**；最后探讨控制工程界广泛应用的**模型预测控制（Model Predictive Control, MPC）**。

---

## 9.1 重新规划与滚动时域 (Receding Horizon Planning)

在离线规划中，智能体预先计算出一个全局策略 $\pi: \mathcal{S} \to \mathcal{A}$，在线时仅需查表执行动作。而在在线规划中，智能体在每个时间步 $t$：
1. 观察当前所处的真实环境状态 $s_t$；
2. 启动在线规划器，以 $s_t$ 为根节点展开有限前瞻深度的未来推演，求解当前时刻的最优行动 $a_t^*$；
3. 将该动作 $a_t^*$ 施加于真实物理系统；
4. 环境转移至新状态 $s_{t+1}$，智能体**抛弃旧的前瞻树，在下一个时间步重复上述规划流程**。

这一不断“规划-执行第一步-滚动向前”的循环机制被称为**滚动时域控制（Receding Horizon Control）**。其核心优势在于：
- **算力精确投射**：将有限的计算资源完全聚焦在当前最可能访问到的局部状态空间，彻底免受高维空间中绝大多数无关状态的干扰；
- **自适应抗扰**：即使物理系统受到突发扰动偏离了预期轨迹，下一时间步的重新规划能够天然纠偏。

---

## 9.2 前瞻搜索 (Forward Search)

最基础的在线规划形式是**深度受限的前瞻搜索（Forward Search）**。

从当前状态 $s$ 出发，搜索树按“动作选择-状态转移”交替展开至指定深度 $d$：
- 动作节点评估其所有可能后继状态的期望效用；
- 状态节点在其候选动作中取最大值（$\max$）；
- 当搜索树到达最大深度 $d$ 时，调用预设的基础评估函数或启发式函数 $U(s)$ 作为叶节点截断收益。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_9_1.png" alt="基础前瞻树搜索拓扑" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.1：包含 3 个状态与 2 个动作的简单前瞻搜索树结构。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_9_2.png" alt="六边形世界深度前瞻演进" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.2：在六边形世界中分别设置前瞻深度为 1 与深度为 2 展开的搜索前沿。</p>
  </div>
</div>

```julia
# 深度受限前瞻搜索算法实现 (来自官方 Julia 算法实现)
struct RolloutLookahead
	𝒫 # problem
	π # rollout policy
	d # depth
end

randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)

function rollout(𝒫, s, π, d)
    ret = 0.0
    for t in 1:d
        a = π(s)
        s, r = randstep(𝒫, s, a)
        ret += 𝒫.γ^(t-1) * r
    end
    return ret
end

function (π::RolloutLookahead)(s)
	U(s) = rollout(π.𝒫, s, π.π, π.d)
    return greedy(π.𝒫, U, s).a
end
####################
```

---

## 9.3 稀疏采样与分支限制 (Sparse Sampling)

在随机 MDP 中，从某状态 $s$ 采取行动 $a$ 后可能转移到的后继状态集合可能极大（甚至在连续状态下是无限的）。这导致搜索树的分支因子发生爆炸。

**稀疏采样（Sparse Sampling, Kearns, Mansour & Ng, 2002）**通过蒙特卡洛抽样严格约束分支度：对于每一个动作分支，算法不枚举所有可能的后继状态，而是**仅利用转移生成器随机独立抽取固定数量 $m$ 个下一状态样本**：
$$
\hat{Q}(s, a) = R(s, a) + \frac{\gamma}{m} \sum_{i=1}^m \hat{U}(s^{(i)})
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_9_3.png" alt="稀疏采样树" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.3：在六边形世界中设置采样分支度 $m=10$ 构建的稀疏采样搜索树。树的总节点规模完全独立于物理状态空间大小。</p>
</div>

**理论突破**：Kearns 等人严格证明：为了保证近最优解，所需的采样分支度 $m$ 与深度 $d$ **完全独立于环境的状态空间规模 $|\mathcal{S}|$**！这意味着即使在无限状态空间中，稀疏采样依然具备多项式样本复杂度的理论保证。

---

## 9.4 蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS)

稀疏采样对所有动作分支均采用均匀无偏的采样，当动作空间较大时仍会浪费大量算力在明显劣质的分支上。**蒙特卡洛树搜索（MCTS）**通过非对称建树，将计算资源高度聚焦于最具潜力的优秀候选路径。

MCTS 在每一个在线决策周期内执行 $n_{\text{sim}}$ 次蒙特卡洛模拟，每次模拟严密包含以下**四大递进步骤**：

```
       1. 选择 (Selection) ───▶ 2. 扩展 (Expansion) ───▶ 3. 模拟 (Simulation) ───▶ 4. 反向传播 (Backpropagation)
         根据 UCB1 选分支        创建新叶节点             快速随机推演至终止       沿路径回传更新 Q 与访问计数 N
```

1. **选择阶段（Selection）**：从根节点出发，如果当前节点的所有动作均已被尝试过，则利用**上限置信区间准则（Upper Confidence Bound for Trees, UCT / UCB1）**选择最优动作分支，直至遇到未完全展开的节点：
   $$
   a^* = \arg\max_{a} \left[ Q(s, a) + c \sqrt{\frac{\log N(s)}{N(s, a)}} \right]
   $$
   式中第一项 $Q(s, a)$ 鼓励**利用（Exploitation）**已知的高回报分支；第二项鼓励**探索（Exploration）**访问次数较少的分支，参数 $c$ 调节探索力度；
2. **扩展阶段（Expansion）**：为当前选中的节点添加一个新生成的子节点；
3. **模拟/推演阶段（Simulation / Rollout）**：从新节点出发，采用计算开销极轻的快速默认策略（如完全均匀随机动作或简单启发式）快速模拟向前推进，直至达到深度截断或环境终止状态，获得推演累积回报 $q$；
4. **反向传播阶段（Backpropagation）**：将本次推演获得的数值回报 $q$ 沿着本次搜索树的访问路径反向回传，更新沿途所有被遍历动作节点的累计效用 $Q(s, a)$ 并将访问计数递增：$N(s, a) \leftarrow N(s, a) + 1$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_9_4.png" alt="2048 棋局的 MCTS 搜索树" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.4：在 2048 经典益智数字棋局中执行 100 次 MCTS 模拟后生成的非对称搜索树拓扑。树的高深分支自动偏向高质量移动方向。</p>
</div>

```julia
# 蒙特卡洛树搜索核心算法实现 (包含 UCB1 选择与 Rollout 推演)
struct ForwardSearch
    𝒫 # problem
    d # depth
    U # value function at depth d
end

function forward_search(𝒫, s, d, U)
    if d ≤ 0
        return (a=nothing, u=U(s))
    end
    best = (a=nothing, u=-Inf)
    U′(s) = forward_search(𝒫, s, d-1, U).u
    for a in 𝒫.𝒜
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::ForwardSearch)(s) = forward_search(π.𝒫, s, π.d, π.U).a
####################
```

---

## 9.5 启发式搜索与实时动态规划 (Heuristic Search & RTDP)

如果智能体预先掌握了一个**可采纳的（Admissible）启发式上界函数**（满足 $U_{\text{heur}}(s) \ge U^*(s)$），可以利用该上界指导搜索并系统性剪枝。

### 实时动态规划 (Real-Time Dynamic Programming, RTDP)
**RTDP（Barto, Bradtke & Singh, 1995）**是在线启发式搜索的典范：
1. 从当前真实状态出发，执行一条前向试错模拟轨迹；
2. 沿途仅对**当前轨迹所访问到的状态**立即执行贝尔曼最优性备份更新；
3. 轨迹终止后重新从当前状态发射新轨迹，反复压紧上界。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_9_5.png" alt="启发式搜索模拟轮数演进" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.5：启发式搜索随着模拟轮次增加（5 次 vs 10 次）的贪心包络收敛态势。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_9_6.png" alt="贪心包络几何" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.6：六边形世界中不同状态触发的贪心包络覆盖范围可视化。</p>
  </div>
</div>

```julia
# 启发式前瞻搜索算法实现
struct BranchAndBound
    𝒫   # problem
    d   # depth
    Ulo # lower bound on value function at depth d
    Qhi # upper bound on action value function
end

function branch_and_bound(𝒫, s, d, Ulo, Qhi)
    if d ≤ 0
        return (a=nothing, u=Ulo(s))
    end
    U′(s) = branch_and_bound(𝒫, s, d-1, Ulo, Qhi).u
    best = (a=nothing, u=-Inf)
    for a in sort(𝒫.𝒜, by=a->Qhi(s,a), rev=true)
        if Qhi(s, a) < best.u
            return best # safe to prune
        end
        u = lookahead(𝒫, U′, s, a)
        if u > best.u
            best = (a=a, u=u)
        end
    end
    return best
end

(π::BranchAndBound)(s) = branch_and_bound(π.𝒫, s, π.d, π.Ulo, π.Qhi).a
####################
```

---

## 9.6 模型预测控制 (Model Predictive Control, MPC)

在连续工程控制领域，**模型预测控制（Model Predictive Control, MPC）**是在线规划的最核心实践支柱。

在 MPC 框架中：
1. 建立系统连续微分动力学方程模型 $\dot{x} = f(x, u)$；
2. 在每个控制步，求解一个跨越未来时间窗口 $T_{\text{horizon}}$ 的数值轨迹轨迹优化命题：
   $$
   \min_{u_{0:H-1}} \quad \sum_{t=0}^{H-1} \ell(x_t, u_t) + \ell_f(x_H)
   $$
3. 将优化求解出的第一拍控制信号 $u_0^*$ 立即输入给电机或执行器；
4. 下一控制时刻测量真实系统状态偏差，重新对齐并再次滚动求解。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_9_9.png" alt="MPC 连续时域轨迹滚动求解" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 9.7：模型预测控制（MPC）在连续倒立摆系统中随时间推进不断重新规划前向轨迹的动态全景。</p>
</div>

---

## 9.7 本章小结 (Summary)

- **计算范式的解耦**：在线规划摆脱了全局离线存储的维度枷锁，通过将算力集中投射于当前状态的局部前瞻，实现了高维与未知环境下的实时自适应决策；
- **稀疏采样的复杂度解构**：通过证明有限蒙特卡洛分支度与物理状态空间大小无关，为随机环境前瞻搜索奠定了样本理论极限；
- **MCTS 的非对称平衡美感**：通过选择、扩展、模拟与回传四步循环，巧妙融合了 UCB1 探索-利用平衡与轻量级蒙特卡洛推演，成为了大状态空间树搜索的行业标准；
- **启发式与控制工程交融**：RTDP 与模型预测控制（MPC）证明了通过滚动时域持续校正误差，能够在现实不确定物理世界中获得极高鲁棒性。

---

## 9.8 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 9.1 (Exercise 9.1)
**题目**：在确定性前瞻搜索树中，设每个状态具有 $|\mathcal{A}|$ 个可选动作。展开至固定深度 $d$ 时，树的叶节点总数是多少？若评估每个叶节点需要调用一次启发式函数，总计算时间复杂度是多少？

**详细解答**：
在确定性环境中，执行动作转移至唯一的确定后继状态：
- 根节点在第 0 层，有 1 个状态；
- 第 1 层有 $|\mathcal{A}|$ 个状态；
- 第 $t$ 层有 $|\mathcal{A}|^t$ 个状态；
- 到达第 $d$ 层时，**叶节点总数为 $|\mathcal{A}|^d$**；
总时间复杂度由遍历整棵树的节点总数决定：
$$
\sum_{t=0}^d |\mathcal{A}|^t = \frac{|\mathcal{A}|^{d+1} - 1}{|\mathcal{A}| - 1} = \mathbf{O(|\mathcal{A}|^d)}
$$
其开销随搜索深度 $d$ 呈指数级增长。

---

### 习题 9.2 (Exercise 9.2)
**题目**：在 MCTS 的 UCB1 动作选择公式中：
$$
a^* = \arg\max_{a} \left[ Q(s, a) + c \sqrt{\frac{\log N(s)}{N(s, a)}} \right]
$$
若探索常数设置过大（$c \to \infty$）或设置过小（$c = 0$），分别会导致搜索树呈现何种病态形态？

**详细解答**：
1. **当 $c = 0$ 时（完全放弃探索）**：
   - 算法退化为纯贪心选择。一旦某个动作在初始几次随机推演中偶然获得较高分数，算法将持续死板地重复遍历该动作，完全不再尝试其他潜在更优分支，极易深陷局部次优解；
2. **当 $c \to \infty$ 时（过度极端探索）**：
   - 探索惩罚项占据绝对支配地位。算法会盲目追求使所有动作分支的访问次数 $N(s, a)$ 强制保持绝对均等，退化为无信息的均匀宽度优先搜索，彻底丧失了将算力非对称倾斜至高回报区域的核心优势。

---

### 习题 9.3 (Exercise 9.3)
**题目**：在稀疏采样中，为什么对每个动作分支抽取的样本数量 $m$ 能够与真实状态空间大小 $|\mathcal{S}|$ 完全无关？简述其理论依据。

**详细解答**：
理论依据来自大数定律与**霍夫丁不等式（Hoeffding's Inequality）**：
蒙特卡洛经验均值对真实期望值的逼近误差界 $\epsilon$，仅取决于**样本数量 $m$ 以及单步效用值的有界范围**，即：
$$
P\left( \left| \frac{1}{m} \sum_{i=1}^m U(s^{(i)}) - \mathbb{E}[U(s')] \right| \ge \epsilon \right) \le 2 \exp\left( -2 m \epsilon^2 / (U_{\max} - U_{\min})^2 \right)
$$
该集中不等式的右端项完全不包含底层总状态数 $|\mathcal{S}|$。因此，只要样本量 $m$ 满足统计精度要求，估算误差就能被有效控制。

---

### 习题 9.4 (Exercise 9.4)
**题目**：在六边形世界中，若当前处于状态 $s$，动作集合包含 6 个方向。若采用 MCTS 进行 50 次模拟。在模拟达到第 10 次时，某个动作 $a_1$ 已被尝试 2 次且累计回报为 4.0；另一个动作 $a_2$ 尝试 1 次且累计回报为 3.0。已知根节点总访问次数为 10，探索常数 $c = \sqrt{2}$。求此时 UCB1 优先选择哪个动作？

**详细解答**：
分别计算两动作当前的平均回报与探索项：
1. **动作 $a_1$**：
   - 平均效用：$Q(s, a_1) = 4.0 / 2 = 2.0$；
   - 探索项：$\sqrt{2} \times \sqrt{\frac{\log 10}{2}} = 1.414 \times \sqrt{\frac{2.3026}{2}} = 1.414 \times \sqrt{1.1513} = 1.414 \times 1.073 = 1.517$；
   - 总分数：$2.0 + 1.517 = \mathbf{3.517}$；
2. **动作 $a_2$**：
   - 平均效用：$Q(s, a_2) = 3.0 / 1 = 3.0$；
   - 探索项：$\sqrt{2} \times \sqrt{\frac{\log 10}{1}} = 1.414 \times \sqrt{2.3026} = 1.414 \times 1.5174 = 2.146$；
   - 总分数：$3.0 + 2.146 = \mathbf{5.146}$；
由于 $5.146 > 3.517$，算法将在本次选择中优先选择 **$a_2$** 分支向下扩展。

---

### 习题 9.5 (Exercise 9.5)
**题目**：实时动态规划（RTDP）为什么要求初始启发式函数必须是“可采纳的”（Admissible）？若不可采纳会有何后果？

**详细解答**：
- **可采纳性**保证了初始启发式估计对所有状态均满足 $U_{\text{heur}}(s) \ge U^*(s)$（即乐观高估）；
- 结合贪心动作选择时，这种“面对不确定性时的乐观主义（Optimism in the face of uncertainty）”驱动智能体主动探索那些当前被高估的未知路径；
- 若初始启发式**不可采纳（即低估了真实最优路径的价值）**，最优路径可能会因为初始低估而被贪心选择永久剪枝，导致智能体永远无法访问该路径，最终收敛至次优策略。

---

### 习题 9.6 (Exercise 9.6)
**题目**：在模型预测控制（MPC）中，既然求解的是跨越未来 $H$ 步的最优控制序列 $u_{0:H-1}^*$，为什么智能体只执行第一拍控制信号 $u_0^*$，而不是将这 $H$ 步动作全部执行完毕？

**详细解答**：
1. **现实环境存在非确定性扰动与建模误差**：物理系统在执行动作后的真实后继状态绝不可能与仿真模型完全一致；
2. 若机械地执行全部 $H$ 步开环指令，误差将随时间步指数级积分漂移累积，导致系统严重失稳；
3. **闭环反馈校正**：通过仅执行第一步 $u_0^*$，在下一拍利用传感器重新测量真实状态后再重新优化，赋予了系统极强的闭环反馈抗扰纠偏能力。

---

### 习题 9.7 (Exercise 9.7)
**题目**：简述 MCTS 中推演阶段（Rollout）使用轻量级启发式策略代替完全随机策略的利与弊。

**详细解答**：
- **利（Pros）**：推演轨迹更加贴近合理博弈与控制策略，能够大幅降低叶节点反向传播回报的方差，显著加速树搜索的收敛速度；
- **弊（Cons）**：若启发式规则存在系统性认知盲区（例如遗漏了某种意想不到的反直觉制胜手段），推演可能会产生强烈的评价偏差，导致搜索树受到误导而忽略潜在的最优奇着。

---

### 习题 9.8 (Exercise 9.8)
**题目**：在什么条件下，在线前瞻规划会比离线价值迭代更受青睐？

**详细解答**：
1. 状态空间规模极其庞大（例如围棋状态数 $10^{170}$、复杂机械臂连续空间），无法离线完整求解；
2. 环境模型仅以**黑盒生成模拟器（Simulators）**的形式给出，无法获取显式的离散概率转移矩阵；
3. 智能体仅需在当前遇到的特定局部状态下做出实时决策，无需对全空间所有生僻状态做常备准备。
