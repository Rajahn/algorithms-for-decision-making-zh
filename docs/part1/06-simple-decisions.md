# 第 6 章：简单决策 (Simple Decisions)

在前几章中，我们探讨了如何表示不确定性、如何在观测到证据后更新信念状态，以及如何从历史数据中学习概率图模型。然而，概率推理本身并非最终目的——**智能体的终极使命是利用概率信念在物理世界中采取行动，以最优化其所追求的目标**。

本章系统探讨**单步决策（One-Shot / Simple Decisions）**理论体系。单步决策是指智能体在单一时间切片内进行一次性选择，其采取的行动不会对长时程的未来后续状态演化产生时序复利影响。我们首先介绍**理性偏好公理（Rational Preference Axioms）**与冯·诺依曼-摩根斯坦定理；随后阐述**效用函数（Utility Functions）**的形式化性质与风险态度（风险厌恶、中性与喜好）；接着引出核心的**最大期望效用原理（Maximum Expected Utility Principle, MEU）**；随后将贝叶斯网络扩展为包含决策与价值节点的**决策网络（Decision Networks / 影响图）**；最后推导评估信息搜集经济价值的**信息价值（Value of Information, VOI）**，并简要探讨人类在现实中的认知偏差与前景理论。

---

## 6.1 理性偏好的约束公理 (Constraints on Rational Preferences)

在面对充满不确定性的未来时，智能体需要在多个不同的可能结果之间做出抉择。我们用**彩票（Lottery）**来形式化描述带有概率不确定性的结局组合：
$$
L = [p_1, S_1; \; p_2, S_2; \; \dots; \; p_m, S_m]
$$
表示以概率 $p_i$ 获得结局状态 $S_i$，满足 $p_i \ge 0$ 且 $\sum_{i=1}^m p_i = 1$。

### 冯·诺依曼-摩根斯坦偏好公理 (VNM Axioms, 1944)
为了使智能体的行为被定义为“理性”，其在所有可能彩票集合上的偏好关系 $\succ$ 必须严格满足以下六条公理：

1. **可排序性（Orderability / Completeness）**：对于任意两个彩票 $A$ 与 $B$，必有且仅有 $A \succ B$、$B \succ A$ 或 $A \sim B$ 之一成立；
2. **传递性（Transitivity）**：若 $A \succ B$ 且 $B \succ C$，则必有 $A \succ C$；
3. **连续性（Continuity）**：若 $A \succ B \succ C$，则必存在唯一的概率值 $p \in [0, 1]$，使得在该确定性结果 $B$ 与包含最好最坏结局的复合彩票之间完全无差别：
   $$
   B \sim [p, A; \; (1 - p), C]
   $$
4. **可替代性（Substitutability）**：若智能体对结局 $A$ 与 $B$ 无差别（$A \sim B$），则在任何复杂彩票中将 $A$ 替换为 $B$，所得的新复合彩票与原彩票等价：
   $$
   [p, A; \; (1 - p), C] \sim [p, B; \; (1 - p), C]
   $$
5. **单调性（Monotonicity）**：若智能体偏好结局 $A$ 优于 $B$（$A \succ B$），则当且仅当优质结局 $A$ 的发生概率更高时，该彩票更受偏好：
   $$
   p > q \iff [p, A; \; (1 - p), B] \succ [q, A; \; (1 - q), B]
   $$
6. **可分解性（Decomposability）**：多阶段复合彩票可以通过标准的概率乘法与加法规则等价简化为单阶段等价复合彩票。

**冯·诺依曼-摩根斯坦定理**证明：**当且仅当智能体的偏好关系满足上述六大理性公理时，必定存在一个实值实值效用函数 $U(S)$，使得智能体对任意两个彩票的偏好顺序完全等价于它们各自的数学期望效用大小！**

---

## 6.2 效用函数与风险态度 (Utility Functions & Risk Attitudes)

效用函数 $U(S) \in \mathbb{R}$ 将可能的结果状态映射为实数标量，反映了智能体对该结局的主观满意度。

### 6.2.1 效用函数的仿射不变性
效用函数的值并不具备绝对的物理单位。冯·诺依曼证明：效用函数在**正仿射变换（Positive Affine Transformation）**下保持偏好不变性：
$$
\hat{U}(S) = a U(S) + b \quad (a > 0)
$$
对原效用函数乘以任意正数 $a$ 并加上常数 $b$，所得的新函数所诱导的全部决策选择与原函数严格等价。因此，我们可以自由对效用函数进行线性归一化，使其最差结局映射为 0，最优结局映射为 1：
$$
\hat{U}(S) = \frac{U(S) - U_{\min}}{U_{\max} - U_{\min}} \in [0, 1]
$$

### 6.2.2 风险态度分类
当面临涉及财富或数值回报 $x$ 的不确定性彩票时，效用函数的二阶导数曲率直接决定了智能体的**风险态度（Risk Attitudes）**：

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_6_1.png" alt="财富效用函数" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 6.1：典型的财富效用曲线，在局部小金额近似线性，在全局呈现凹函数。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_6_2.png" alt="幂效用函数族" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 6.2：不同参数 $\lambda$ 下的幂效用函数曲线。</p>
  </div>
</div>

1. **风险中性（Risk Neutral）**：效用函数为严格线性 $U(x) = c x$。智能体仅关注数学期望回报，对任何公平赌局持无差别态度；
2. **风险厌恶（Risk Averse）**：效用函数为**严格凹函数（Concave）**，满足 $U''(x) < 0$（边际效用递减）。
   - **确定性等价值（Certainty Equivalent, CE）**：使智能体在“确定性获得金额 $\text{CE}$”与“不确定性彩票”之间无差别的现金数值，必严格小于彩票的数学期望：
     $$
     \text{CE} < \mathbb{E}[X]
     $$
   - **风险溢价（Risk Premium）**：$\text{RP} = \mathbb{E}[X] - \text{CE} > 0$，代表智能体为了规避风险而自愿放弃的期望收益（这也是商业保险能够成立的经济学根基）；
3. **风险喜好（Risk Seeking）**：效用函数为**严格凸函数（Convex）**，满足 $U''(x) > 0$，其确定性等价值严格大于彩票期望。

常用的参数化效用函数是**幂效用函数（Power Utility）**：
$$
U(x) = \begin{cases} \frac{x^{1 - \lambda} - 1}{1 - \lambda} & (\lambda \neq 1) \\ \log x & (\lambda = 1) \end{cases}
$$
式中参数 $\lambda > 0$ 正好衡量了智能体的相对风险厌恶系数。

---

## 6.3 效用诱导 (Utility Elicitation)

在工程实践中，如何获知人类专家或用户的真实效用函数？最经典的标准评估程序是**标准博弈法（Standard Gamble Method）**：

1. 选定环境中最差的灾难性结果 $S_-$，赋予效用 $U(S_-) = 0$；
2. 选定环境中最好的理想结果 $S_+$，赋予效用 $U(S_+) = 1$；
3. 对于任意待评估的中间结果 $S$：向决策者提出一个选择：
   - 方案一：确定性获得中间结果 $S$；
   - 方案二：参与一个彩票，以概率 $p$ 获得最优结局 $S_+$，以概率 $1 - p$ 获得最差结局 $S_-$；
4. 动态调整概率 $p$，直到决策者对方案一与方案二感到完全无差别。根据期望效用等价原理，该中间状态的真实效用即精确等于该平衡概率：
   $$
   U(S) = p \cdot U(S_+) + (1 - p) \cdot U(S_-) = p
   $$

---

## 6.4 最大期望效用原理 (Maximum Expected Utility Principle, MEU)

设智能体拥有可选离散行动集合 $\mathcal{A}$，采取行动 $a \in \mathcal{A}$ 后可能导致不同的客观世界状态 $s \in \mathcal{S}$，其条件概率为 $P(s \mid a)$。

**最大期望效用原理（MEU）**指出，理性的智能体应当选择能够最大化期望效用的行动：
$$
a^* = \arg\max_{a \in \mathcal{A}} \mathbb{E}[U(a)] = \arg\max_{a \in \mathcal{A}} \sum_{s} P(s \mid a) U(s)
$$

---

## 6.5 决策网络与影响图 (Decision Networks)

为了将贝叶斯网络拓展为能够直接支持决策制定的计算图模型，罗纳德·霍华德（Ronald Howard）引入了**影响图（Influence Diagrams）**，在人工智能领域通常被称为**决策网络（Decision Networks）**。

决策网络在贝叶斯网络的基础上引入了两种全新的专用节点类型：
1. **偶然状态节点（Chance Nodes，圆形/椭圆形）**：代表服从概率分布的环境随机变量，遵循标准贝叶斯网络条件概率表；
2. **决策节点（Decision Nodes，矩形）**：代表受智能体直接控制的可选行动变量，其入边代表**做出该决策时智能体已知的信息线索**；
3. **效用节点（Utility Nodes，菱形）**：代表智能体获得的数值回报，其父节点为其效用评估所直接依赖的环境状态或行动。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_6_dn1.png" alt="基础决策网络" style="max-width: 250px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 6.3：示例 6.4 的雨伞决策网络。气象预报作为行动的信息输入。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_6_dn2.png" alt="包含测试动作的高级决策网络" style="max-width: 270px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 6.4：示例 6.5 包含预先测试动作的决策网络架构。</p>
  </div>
</div>

```julia
# 决策网络数据结构定义 (来自官方 Julia 算法实现)
struct SimpleProblem
    bn::BayesianNetwork
    chance_vars::Vector{Variable}
    decision_vars::Vector{Variable}
    utility_vars::Vector{Variable}
    utilities::Dict{Symbol, Vector{Float64}}
end

function solve(𝒫::SimpleProblem, evidence, M)
    query = [var.name for var in 𝒫.utility_vars]
    U(a) = sum(𝒫.utilities[uname][a[uname]] for uname in query)
    best = (a=nothing, u=-Inf)
    for assignment in assignments(𝒫.decision_vars)
        evidence = merge(evidence, assignment)
        ϕ = infer(M, 𝒫.bn, query, evidence)
        u = sum(p*U(a) for (a, p) in ϕ.table)
        if u > best.u
            best = (a=assignment, u=u)
        end
    end
    return best
end
####################

# 决策网络精确求解与动作期望效用评估算法
function value_of_information(𝒫, query, evidence, M)
    ϕ = infer(M, 𝒫.bn, query, evidence)
    voi = -solve(𝒫, evidence, M).u
    query_vars = filter(v->v.name ∈ query, 𝒫.chance_vars)
    for o′ in assignments(query_vars)
        oo′ = merge(evidence, o′)
        p = ϕ.table[o′]
        voi += p*solve(𝒫, oo′, M).u
    end
    return voi
end
####################
```

求解决策网络的最优决策规则，可以直接通过变量消除算法将效用节点当作特殊的因子，对非信息偶然变量进行求和边际化，从而评估每个备选行动的期望效用。

---

## 6.6 信息价值 (Value of Information, VOI)

在现实世界中，智能体在做出最终决策之前，往往可以通过支付一定成本来获取额外的观测信息（例如气象卫星扫描、医疗核磁共振检查、深空探测车传感器采样）。我们究竟应当为这些新信息支付多少代价？

这由**信息价值（Value of Information, VOI）**（又称完全信息价值 VPI）统领。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_6_voi.png" alt="信息价值决策网络" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 6.5：用于计算传感器观测节点信息价值的决策网络结构扩展。</p>
</div>

### 6.6.1 数学定义
设在没有额外观测时，智能体基于当前先验采取最优行动的基准期望效用为：
$$
\text{EU}^* = \max_{a} \sum_{s} P(s) U(s, a)
$$
若我们有机会在决策前预先确切获知变量 $O$ 的实际观测值 $o$：
- 在获知 $O=o$ 后，后验概率更新为 $P(s \mid o)$，智能体调整行动以最大化后验期望效用：
  $$
  \text{EU}^*(o) = \max_{a} \sum_{s} P(s \mid o) U(s, a)
  $$
- 在真正执行观测前，变量 $O$ 本身的取值是未知的，其发生概率为 $P(o)$。因此预先获知该信息的**总体期望后效用**为：
  $$
  \mathbb{E}_O[\text{EU}^*(O)] = \sum_{o} P(o) \max_{a} \left[ \sum_{s} P(s \mid o) U(s, a) \right]
  $$

**完全信息价值（VOI）**定义为获取信息前后的期望效用增量：
$$
\text{VOI}(O) = \mathbb{E}_O[\text{EU}^*(O)] - \text{EU}^*
$$

### 6.6.2 重要定理：信息的非负性
**定理**：对于任意理性智能体与任意观测变量 $O$，其信息价值**恒大于等于零**：
$$
\text{VOI}(O) \ge 0
$$
**数学直觉**：由于先求最大值后求期望恒大于等于先求期望后求最大值（凸函数詹森不等式），智能体永远可以选择“忽略新信息而坚持原方案”，新信息只赋予了智能体更灵活地根据现实因地制宜调整行动的自由度，绝不可能使最优期望效用受损。

---

## 6.7 人类非理性与行为决策 (Irrationality & Prospect Theory)

尽管冯·诺依曼期望效用理论为人工智能提供了优雅的数学公理体系，但行为经济学的大量实证研究发现，**人类在现实中的实际决策频繁系统性地违背 VNM 理性公理**：

1. **阿莱悖论（Allais Paradox, 1953）**：通过精心设计的两组具有相同概率差值的彩票对决，实验证明人类普遍更渴望“确定性（Certainty Effect）”，系统性破坏了**可替代性公理**；
2. **埃尔斯伯格悖论（Ellsberg Paradox, 1961）**：人类对已知概率的风险与完全未知概率的“模糊性（Ambiguity）”存在强烈的非对称偏好（模糊厌恶），破坏了先验概率测度的可加性；
3. **框架效应与前景理论（Prospect Theory, Kahneman & Tversky, 1979）**：
   - **损失厌恶（Loss Aversion）**：同等金额的损失所带来的心理痛苦，约为同等金额收益所带来愉悦感的 2 至 2.5 倍；
   - **非线性概率加权**：人类普遍严重高估罕见小概率事件（如空难、买彩票中大奖），同时低估中等偏高概率事件。

理解人类的系统性非理性偏差，对于设计辅助人类决策的协作型人工智能系统、避免算法诱导用户做出危害自身长期福祉的非理性行为具有重大意义。

---

## 6.8 本章小结 (Summary)

- **理性公理基石**：冯·诺依曼-摩根斯坦六大公理确保了偏好关系能够被期望效用严格量化；
- **效用与风险度量**：效用函数具有正仿射不变性；效用函数的二阶凹凸性深刻刻画了风险中性、风险厌恶与风险喜好的行为根源；
- **最大期望效用准则**：在不确定性环境下，智能体的最优决策是在当前概率信念切片下追求期望效用的极大化；
- **决策网络图模型**：通过在贝叶斯网络中引入方形决策节点与菱形效用节点，构建了规范化的单步决策计算引擎；
- **信息价值与非负性**：VOI 量化了收集新观测能够带来的效用跃升期望，非负性定理证明了额外信息在理性视角下的绝对正向收益。

---

## 6.9 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 6.1 (Exercise 6.1)
**题目**：设效用函数 $U(s)$ 具有有限的最大值 $\bar{U}$ 与有限的最小值 $\underline{U}$。写出其对应的归一化效用函数 $\hat{U}(s)$，使其输出严格位于 $[0, 1]$ 区间。

**详细解答**：
利用效用函数的正仿射变换性质，构造线性变换 $\hat{U}(s) = a U(s) + b$。
令 $\hat{U}(\underline{s}) = 0$ 且 $\hat{U}(\bar{s}) = 1$：
$$
\hat{U}(s) = \mathbf{\frac{U(s) - \underline{U}}{\bar{U} - \underline{U}}}
$$
由于 $\bar{U} - \underline{U} > 0$，这是一个保序正仿射变换，诱导完全相同的决策偏好。

---

### 习题 6.2 (Exercise 6.2)
**题目**：已知结局偏好关系为 $A \succeq C \succeq B$，各结局效用分别为 $U(A) = 450$，$U(B) = -150$，$U(C) = 60$。求一个定义在最优结局 $A$ 与最差结局 $B$ 之间的标准彩票 $[p, A; \; (1-p), B]$，使得智能体在该彩票与确定性中间结局 $C$ 之间无差别。

**详细解答**：
根据期望效用等价无差别条件：
$$
U(C) = p \cdot U(A) + (1 - p) \cdot U(B)
$$
代入效用数值：
$$
60 = 450 p - 150 (1 - p) = 450 p - 150 + 150 p = 600 p - 150
$$
解一元一次方程：
$$
600 p = 210 \implies p = \frac{210}{600} = \mathbf{0.35}
$$
故当且仅当以 $p = 0.35$ 的概率获得 $A$、以 $0.65$ 的概率获得 $B$ 时，智能体与确定获得 $C$ 完全无差别。

---

### 习题 6.3 (Exercise 6.3)
**题目**：设效用函数定义在三结局上：$U(A) = 5$，$U(B) = 20$，$U(C) = 0$。智能体面临两种选择：彩票 $L_1 = [0.5, A; \; 0.5, B]$ 与彩票 $L_2 = [0.8, A; \; 0.2, C]$。求智能体应当偏好哪一个彩票？

**详细解答**：
分别计算两彩票的期望效用：
- 彩票 1：$\mathbb{E}[U(L_1)] = 0.5 \times U(A) + 0.5 \times U(B) = 0.5 \times 5 + 0.5 \times 20 = 2.5 + 10 = \mathbf{12.5}$；
- 彩票 2：$\mathbb{E}[U(L_2)] = 0.8 \times U(A) + 0.2 \times U(C) = 0.8 \times 5 + 0.2 \times 0 = \mathbf{4.0}$；
由于 $\mathbb{E}[U(L_1)] = 12.5 > \mathbb{E}[U(L_2)] = 4.0$，根据最大期望效用原理，智能体严格偏好 **$L_1 \succ L_2$**。

---

### 习题 6.4 (Exercise 6.4)
**题目**：证明幂效用函数 $U(x) = \frac{x^{1 - \lambda} - 1}{1 - \lambda}$ 在 $x > 0$、$\lambda > 0$ 且 $\lambda \neq 1$ 时严格表现为风险厌恶（Risk Averse）。

**详细解答**：
智能体表现为风险厌恶的充要条件是效用函数二阶导数严格为负（严格凹函数，即 $U''(x) < 0$）。
对幂效用函数求一阶导数：
$$
U'(x) = \frac{d}{dx} \left[ \frac{x^{1 - \lambda} - 1}{1 - \lambda} \right] = \frac{(1 - \lambda) x^{-\lambda}}{1 - \lambda} = x^{-\lambda}
$$
求二阶导数：
$$
U''(x) = \frac{d}{dx} \left[ x^{-\lambda} \right] = -\lambda x^{-\lambda - 1}
$$
由于已知 $x > 0$ 且 $\lambda > 0$，故 $x^{-\lambda - 1} > 0$ 恒成立，因此：
$$
U''(x) = -\lambda x^{-\lambda - 1} < 0 \quad (\forall x > 0)
$$
二阶导数在定义域内处处严格小于零，函数严格凹，因此该效用函数处处严格表现为**风险厌恶**。

---

### 习题 6.5 (Exercise 6.5)
**题目**：使用示例 6.3 中的下雨决策参数：天气 $W \in \{\text{rain}, \text{sun}\}$，行动 $A \in \{\text{bring}, \text{leave}\}$。已知效用矩阵 $U(\text{rain}, \text{bring}) = 2$，$U(\text{sun}, \text{bring}) = 0$，$U(\text{rain}, \text{leave}) = 0$，$U(\text{sun}, \text{leave}) = 3$。若气象预报为晴天（forecast = sun），已知在此预报下后验概率为 $P(\text{rain} \mid \text{sun}) = 0.1$，$P(\text{sun} \mid \text{sun}) = 0.9$。求在此预报下带伞与不带伞的期望效用。

**详细解答**：
在给定预报为晴天的后验概率下分别求期望效用：
1. **带伞行动（$A = \text{bring}$）**：
   $$
   \mathbb{E}[U(\text{bring}) \mid \text{forecast}=\text{sun}] = P(\text{rain} \mid \text{sun}) U(\text{rain}, \text{bring}) + P(\text{sun} \mid \text{sun}) U(\text{sun}, \text{bring}) = 0.1 \times 2 + 0.9 \times 0 = \mathbf{0.2}
   $$
2. **不带伞行动（$A = \text{leave}$）**：
   $$
   \mathbb{E}[U(\text{leave}) \mid \text{forecast}=\text{sun}] = P(\text{rain} \mid \text{sun}) U(\text{rain}, \text{leave}) + P(\text{sun} \mid \text{sun}) U(\text{sun}, \text{leave}) = 0.1 \times 0 + 0.9 \times 3 = \mathbf{2.7}
   $$
结论：在预报晴天时，不带伞的期望效用（2.7）远大于带伞（0.2），最优决策是不带伞。

---

### 习题 6.6 (Exercise 6.6)
**题目**：设小狗处于饥饿状态 $H \in \{0, 1\}$，先验 $P(H=1) = 0.5$。行动 $F \in \{0, 1\}$ 代表是否喂食。小狗可能会摇尾巴 $W \in \{0, 1\}$，已知 $P(W=1 \mid H=0) = 0.8$，$P(W=1 \mid H=1) = 0.2$。效用矩阵为：若小狗饥饿且喂食 $U(H=1, F=1) = 10$，饥饿但未喂食 $U(H=1, F=0) = -20$，饱食且喂食 $U(H=0, F=1) = -5$，饱食且未喂食 $U(H=0, F=0) = 5$。若观察到小狗摇尾巴（$W=1$），求最优喂食决策。

**详细解答**：
1. 首先利用贝叶斯法则计算小狗在摇尾巴条件下的饥饿后验概率：
   $$
   P(H=1 \mid W=1) = \frac{P(W=1 \mid H=1)P(H=1)}{P(W=1 \mid H=1)P(H=1) + P(W=1 \mid H=0)P(H=0)} = \frac{0.2 \times 0.5}{0.2 \times 0.5 + 0.8 \times 0.5} = \frac{0.1}{0.1 + 0.4} = 0.2
   $$
   故 $P(H=0 \mid W=1) = 1 - 0.2 = 0.8$。
2. 计算喂食（$F=1$）的条件期望效用：
   $$
   \mathbb{E}[U(F=1) \mid W=1] = 0.2 \times U(H=1, F=1) + 0.8 \times U(H=0, F=1) = 0.2 \times 10 + 0.8 \times (-5) = 2 - 4 = \mathbf{-2}
   $$
3. 计算不喂食（$F=0$）的条件期望效用：
   $$
   \mathbb{E}[U(F=0) \mid W=1] = 0.2 \times U(H=1, F=0) + 0.8 \times U(H=0, F=0) = 0.2 \times (-20) + 0.8 \times 5 = -4 + 4 = \mathbf{0}
   $$
4. 结论：由于 $0 > -2$，当观察到小狗摇尾巴时，最优决策是**不喂食（$F=0$）**。

---

### 习题 6.7 (Exercise 6.7)
**题目**：基于习题 6.6 的模型，假设小狗是否跑向食盆由变量 $R \in \{0, 1\}$ 表示。已知在摇尾巴条件下，不跑向食盆时饥饿概率为 $P(H=1 \mid W=1, R=0) = 0.05$，跑向食盆时饥饿概率为 $P(H=1 \mid W=1, R=1) = 0.80$，跑向食盆的边际概率为 $P(R=1 \mid W=1) = 0.20$。在已经观察到摇尾巴（$W=1$）后，进一步观察小狗是否跑向食盆（$R$）的完全信息价值 $\text{VOI}(R \mid W=1)$ 是多少？

**详细解答**：
1. **未获知 $R$ 时的基准最大期望效用**：
   由习题 6.6 已知：$\text{EU}^*(W=1) = \max(-2, 0) = \mathbf{0}$（此时选择不喂食）。
2. **若观察到 $R=1$（跑向食盆，发生概率 0.2）**：
   - 此时 $P(H=1) = 0.8$，$P(H=0) = 0.2$；
   - 喂食期望：$0.8 \times 10 + 0.2 \times (-5) = 8 - 1 = 7$；
   - 不喂食期望：$0.8 \times (-20) + 0.2 \times 5 = -16 + 1 = -15$；
   - 最优选择喂食，$\text{EU}^*(W=1, R=1) = \mathbf{7}$。
3. **若观察到 $R=0$（未跑向食盆，发生概率 0.8）**：
   - 此时 $P(H=1) = 0.05$，$P(H=0) = 0.95$；
   - 喂食期望：$0.05 \times 10 + 0.95 \times (-5) = 0.5 - 4.75 = -4.25$；
   - 不喂食期望：$0.05 \times (-20) + 0.95 \times 5 = -1.0 + 4.75 = 3.75$；
   - 最优选择不喂食，$\text{EU}^*(W=1, R=0) = \mathbf{3.75}$。
4. **获取信息后的期望效用**：
   $$
   \mathbb{E}_R[\text{EU}^*(W=1, R)] = P(R=1 \mid W=1) \times 7 + P(R=0 \mid W=1) \times 3.75 = 0.2 \times 7 + 0.8 \times 3.75 = 1.4 + 3.0 = \mathbf{4.4}
   $$
5. **计算信息价值**：
   $$
   \text{VOI}(R \mid W=1) = \mathbb{E}_R[\text{EU}^*(W=1, R)] - \text{EU}^*(W=1) = 4.4 - 0 = \mathbf{4.4}
   $$
该观察动作具有显著的正信息价值（4.4 个效用单位），能大幅指导喂食决策。
