# 第 24 章：多智能体推理 (Multiagent Reasoning)

在前面的所有章节中，我们均假设环境中仅存在**单一的决策智能体**。环境中的其他实体要么被简化为静态的物理障碍，要么被建模为服从固定概率分布的随机自然现象。

然而在现实世界的诸多关键场景中，环境中同时活跃着**多个具备自主认知能力与独立利益诉求的决策智能体**。例如在自动驾驶十字路口，其他车辆驾驶员同样在根据你的动作实时调整变道策略；在金融交易市场中，其他机构的高频算法正在试图预测并利用你的挂单行为；在无人机蜂群作战中，敌我双方均在动态推演对手的防空拦截边界。

在此类环境中，环境不再是静态客观的物理介质，而是展现出强烈的**对抗、博弈与策略互动性**。本章正式迈入**博弈论（Game Theory）与多智能体系统（Multiagent Systems）**的殿堂。我们首先形式化定义**策略型博弈（Strategic Games）**；随后深入剖析博弈论的基石——**纳什均衡（Nash Equilibrium）**与零和博弈的**极小极大定理（Minimax Theorem）**；接着探讨**迭代剔除严格劣势策略（IESDS）**；随后引入更符合人类认知行为的**反应模型与认知层级（Level-$k$ Reasoning）**；最后系统阐述**虚拟博弈（Fictitious Play）**与多智能体连续梯度上升博弈动态。

---

## 24.1 策略型博弈形式化 (Strategic Games)

最基础的多智能体决策模型是**策略型博弈（Strategic Form Games / 范式博弈 Normal Form Games）**，用于刻画所有参与者在单一切片内**同时、独立选择行动**的静态决策交互。

一个策略型博弈由三元组 $(\mathcal{I}, \mathcal{A}, \mathcal{U})$ 精确定义：
1. **智能体集合 $\mathcal{I} = \{1, \dots, n\}$**：参与决策的 $n$ 个独立主体；
2. **联合动作空间 $\mathcal{A} = \mathcal{A}_1 \times \dots \times \mathcal{A}_n$**：每个智能体拥有自身可选的动作集合 $\mathcal{A}_i$。一个特定的组合 $\mathbf{a} = (a_1, \dots, a_n) \in \mathcal{A}$ 称为**联合动作剖面（Joint Action Profile）**。常用记号 $\mathbf{a}_{-i} = (a_1, \dots, a_{i-1}, a_{i+1}, \dots, a_n)$ 表示除智能体 $i$ 之外其余所有参与者的联合动作；
3. **效用函数集合 $\mathcal{U} = \{U_1, \dots, U_n\}$**：每个智能体拥有独立的实值效用函数 $U_i(\mathbf{a}) = U_i(a_i, \mathbf{a}_{-i})$，其收益不仅取决于自身的选择 $a_i$，而且强烈取决于对手的选择 $\mathbf{a}_{-i}$。

### 纯策略与混合策略 (Pure & Mixed Strategies)
- **纯策略（Pure Strategy）**：确定性选择某个单一动作 $a_i \in \mathcal{A}_i$；
- **混合策略（Mixed Strategy）**：在动作空间 $\mathcal{A}_i$ 上定义一个概率分布 $\boldsymbol{\pi}_i \in \Delta^{|\mathcal{A}_i|-1}$，以概率 $\pi_i(a_i)$ 随机选择执行动作 $a_i$。在混合策略剖面 $\boldsymbol{\pi} = (\boldsymbol{\pi}_1, \dots, \boldsymbol{\pi}_n)$ 下，智能体 $i$ 的期望效用为：
  $$
  U_i(\boldsymbol{\pi}) = \sum_{\mathbf{a} \in \mathcal{A}} \left( \prod_{j=1}^n \pi_j(a_j) \right) U_i(\mathbf{a})
  $$

```julia
# 策略型博弈核心数据结构定义 (来自官方 Julia 算法实现)
struct SimpleGame
    γ  # discount factor
    ℐ  # agents
    𝒜  # joint action space
    R  # joint reward function
end
####################
```

---

## 24.2 纳什均衡与极小极大定理 (Nash Equilibrium)

在多主体互动中，智能体应当如何预测稳定的博弈结局？约翰·纳什（John Nash, 1950）给出了博弈论最著名的解决方案。

### 24.2.1 最优反应与纳什均衡定义
给定其他对手的策略剖面 $\boldsymbol{\pi}_{-i}$，智能体 $i$ 能够最大化自身期望回报的策略集合称为对该对手局面的**最优反应（Best Response, BR）**：
$$
\text{BR}_i(\boldsymbol{\pi}_{-i}) = \arg\max_{\boldsymbol{\pi}_i} U_i(\boldsymbol{\pi}_i, \boldsymbol{\pi}_{-i})
$$

**纳什均衡（Nash Equilibrium）**定义为一个策略剖面 $\boldsymbol{\pi}^* = (\boldsymbol{\pi}_1^*, \dots, \boldsymbol{\pi}_n^*)$，满足**每个智能体的策略同时互为对方当前策略的最优反应**：
$$
U_i(\boldsymbol{\pi}_i^*, \boldsymbol{\pi}_{-i}^*) \ge U_i(\boldsymbol{\pi}_i, \boldsymbol{\pi}_{-i}^*) \quad (\forall \boldsymbol{\pi}_i \in \Delta^{|\mathcal{A}_i|-1}, \; \forall i \in \mathcal{I})
$$
在纳什均衡状态下，**没有任何单一智能体能够通过单方面单方面改变自身策略而获得更高的期望收益**！这是一个自我施加的战略稳态。

**纳什存在性定理（Nash's Theorem, 1950）**：任何有限参与者、有限纯动作的策略型博弈，**必然存在至少一个混合策略纳什均衡**！

### 24.2.2 两人零和博弈与极小极大定理 (Minimax Theorem)
在利益完全对立的**两人零和博弈（Two-Player Zero-Sum Games）**中，$U_1(\mathbf{a}) + U_2(\mathbf{a}) = 0$ 恒成立。冯·诺依曼（John von Neumann, 1928）证明了著名的**极小极大定理**：
$$
\max_{\boldsymbol{\pi}_1} \min_{\boldsymbol{\pi}_2} U_1(\boldsymbol{\pi}_1, \boldsymbol{\pi}_2) = \min_{\boldsymbol{\pi}_2} \max_{\boldsymbol{\pi}_1} U_1(\boldsymbol{\pi}_1, \boldsymbol{\pi}_2) = V^*
$$
该标量常数 $V^*$ 称为博弈的**博弈值（Value of the Game）**。在零和博弈中，纳什均衡策略等价于最保守稳健的安全策略，可以通过标准的**线性规划（Linear Programming）**在多项式时间内精确求解。

```julia
# 零和博弈线性规划求解纳什均衡算法实现
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities

    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end

    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)

function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)
end

joint(X) = vec(collect(product(X...)))

joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i]*p(a) for a in joint(𝒜))
end
####################
```

---

## 24.3 迭代剔除严格劣势策略 (Iterated Elimination of Dominated Strategies, IESDS)

若某动作 $a_i$ 无论对手采取何种策略组合 $\mathbf{a}_{-i}$，其带来的效用始终严格低于另一个动作 $a_i'$：
$$
U_i(a_i, \mathbf{a}_{-i}) < U_i(a_i', \mathbf{a}_{-i}) \quad (\forall \mathbf{a}_{-i} \in \mathcal{A}_{-i})
$$
则称动作 $a_i$ 是一个**严格劣势策略（Strictly Dominated Strategy）**。

**IESDS 算法机理**：
1. 理性的智能体绝不可能执行严格劣势策略，故可将其从博弈矩阵中永久剔除；
2. 剔除劣势动作后，对手在新矩阵中可能会涌现出新的劣势动作；
3. 反复交替剔除，直至矩阵不可再约减。

在著名的**囚徒困境（Prisoner's Dilemma）**博弈中，“背叛（Defect）”是每个参与者不论对方如何选择均能获益的绝对占优策略。IESDS 能够单调且唯一地直接收敛至双重背叛的纳什均衡点。

---

## 24.4 反应模型与认知层级 (Response Models & Cognitive Hierarchy)

纳什均衡建立在“所有参与者均无限理性且彼此知晓对方无限理性”的极端假设之上。在现实博弈实证实验中，人类普遍表现出有限理性的阶梯认知特征。

### Level-$k$ 思考模型 (Stahl & Wilson, 1994)
- **Level 0 智能体**：完全无战略思考，在动作空间中均匀随机胡乱出招；
- **Level 1 智能体**：假设所有对手均为 Level 0，针对完全随机的对手计算最优反应；
- **Level $k$ 智能体**：假设所有对手均处于更低阶的 Level $k-1$，并计算对应的分层 Softmax 最优反应：
  $$
  P(a_i \mid \text{Level-}k) \propto \exp\left( \lambda \cdot U_i(a_i, \boldsymbol{\pi}_{-i}^{k-1}) \right)
  $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_24_1.png" alt="分层 Softmax 模型在旅行者困境中的响应" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 24.1：分层 Softmax 认知模型在经典旅行者困境（Traveler's Dilemma）中随理性参数 $\lambda$ 变化的概率密度曲面。</p>
</div>

```julia
# 认知层级与分层 Softmax 模型实现
function best_response(𝒫::SimpleGame, π, i)
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    ai = argmax(U, 𝒫.𝒜[i])
    return SimpleGamePolicy(ai)
end
####################
```

---

## 24.5 虚拟博弈与动态演化 (Fictitious Play & Gradient Ascent)

在重复博弈中，智能体如何随时间进化其策略？

### 24.5.1 虚拟博弈 (Fictitious Play, Brown, 1951)
每个智能体假设对手遵循某种固定的平稳混合策略，通过统计对手在历史所有轮次中出招的**经验频数经验分布**来建立对手模型：
$$
\hat{\pi}_{-i}^{(t)}(a) = \frac{N_{-i}^{(t)}(a)}{\sum_{a'} N_{-i}^{(t)}(a')}
$$
并在每一轮中针对该经验对手模型贪心选择单步最优反应动作。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_24_2.png" alt="虚拟博弈在囚徒困境中的收敛" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 24.2：两名虚拟博弈智能体在重复囚徒困境中策略迅速锁定合作与背叛的演化轨迹。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_24_3.png" alt="虚拟博弈在石头剪刀布中的轨道循环" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 24.3：两名虚拟博弈智能体在“石头-剪刀-布”循环博弈中，经验模型环绕中心纳什均衡点反复盘旋收缩的动态螺旋轨道。</p>
  </div>
</div>

### 24.5.2 连续梯度上升博弈动态 (Gradient Ascent in Games)
每个智能体直接通过策略梯度最大化自身期望效用：$\boldsymbol{\theta}_i \leftarrow \boldsymbol{\theta}_i + \alpha \nabla_{\boldsymbol{\theta}_i} U_i(\boldsymbol{\theta}_i, \boldsymbol{\theta}_{-i})$。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_24_4.png" alt="梯度上升智能体在猜硬币游戏中的向量场" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 24.4：两名纯独立梯度上升智能体在非凸猜硬币游戏（Matching Pennies）中展现出的闭合极限环周期震荡轨迹。</p>
</div>

```julia
# 虚拟博弈算法实现
function softmax_response(𝒫::SimpleGame, π, i, λ)
    𝒜i = 𝒫.𝒜[i]
    U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
    return SimpleGamePolicy(ai => exp(λ*U(ai)) for ai in 𝒜i)
end
####################
```

---

## 24.6 本章小结 (Summary)

- **多主体交互的本质跃迁**：环境状态转移与回报由全部参与者的联合动作共同决定，个体优化与全局均衡发生深刻碰撞；
- **纳什均衡的划时代地位**：定义了博弈论最核心的单方面不可偏离战略稳态，零和博弈中等价于安全极小极大线性规划；
- **劣势策略剔除与严格简化**：IESDS 提供了从纯理性公理出发快速压缩状态矩阵的数学利器；
- **有限理性阶梯**：Level-$k$ 认知模型以优雅的分层结构精确捕捉了人类面对博弈时的真实思维深度；
- **动态学习与循环震荡**：虚拟博弈与梯度上升展示了多主体同时自适应学习时可能涌现出的复杂混沌、极限环螺旋与收敛动力学。

---

## 24.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 24.1 (Exercise 24.1)
**题目**：考虑经典“协调博弈”（Coordination Game），两名参与者分别选择去左侧咖啡馆（L）或右侧咖啡馆（R）。若两人选择相同地点则各自获得效用 10，若选择不同地点则各获得 0。写出该博弈的收益矩阵，并求其所有的纯策略与混合策略纳什均衡。

**详细解答**：
1. 收益矩阵为 $2 \times 2$ 矩阵：
   - 动作剖面 $(L, L)$：效用 $(10, 10)$；
   - 动作剖面 $(L, R)$：效用 $(0, 0)$；
   - 动作剖面 $(R, L)$：效用 $(0, 0)$；
   - 动作剖面 $(R, R)$：效用 $(10, 10)$；
2. **纯策略纳什均衡**：
   - $(L, L)$：若对方选 $L$，自身选 $L$ 获得 10，改选 $R$ 获得 0，无法偏离；
   - $(R, R)$：同理，双方均选择 $R$ 构成稳态；
   故纯策略纳什均衡共有 2 个：**$(L, L)$ 与 $(R, R)$**；
3. **混合策略纳什均衡**：
   设智能体 1 以概率 $p$ 选择 $L$，智能体 2 以概率 $q$ 选择 $L$。
   为使智能体 1 在 $L$ 与 $R$ 之间无差别：
   $$
   U_1(L, q) = 10 q + 0 (1 - q) = 10 q, \quad U_1(R, q) = 0 q + 10 (1 - q) = 10(1 - q)
   $$
   令 $10 q = 10(1 - q) \implies 2q = 1 \implies q^* = \mathbf{0.5}$；
   对称地解得 $p^* = \mathbf{0.5}$。
   故存在第 3 个混合策略纳什均衡：**双方均以 $50\%$ 的概率随机抛硬币选择 $L$ 或 $R$**（此时期望效用仅为 5.0，劣于协同纯均衡）。

---

### 习题 24.2 (Exercise 24.2)
**题目**：在“剪刀-石头-布”零和博弈中，证明唯一的纳什均衡是双方均以完全相等的概率 $\frac{1}{3}$ 均匀随机出招，且该博弈的博弈值 $V^* = 0$。

**详细解答**：
收益矩阵为对称斜对称矩阵（获胜收益 +1，打平 0，失败 -1）。
设对手策略为 $\mathbf{q} = [q_1, q_2, q_3]^\top$。
若智能体选择策略 $\mathbf{p} = [1/3, 1/3, 1/3]^\top$：
对对手出石头、剪刀或布中任意单一招式，其期望收益为：
$$
U_1(\mathbf{p}, \text{石头}) = \frac{1}{3}(0) + \frac{1}{3}(-1) + \frac{1}{3}(1) = 0
$$
同理对任意动作期望回报恒为 0。因此对手无法通过单方面改变策略获得超过 0 的收益；对称地，对手同样采用均匀分布时智能体也无法获得超过 0 的收益。
根据极小极大定理，**双方均匀随机出招是唯一的纳什均衡，博弈值严格等于 $V^* = 0$**。

---

### 习题 24.3 (Exercise 24.3)
**题目**：简述“囚徒困境”中为什么个体理性的最优选择最终却无可挽回地导致了集体非理性的悲剧结果。

**详细解答**：
在囚徒困境中，对于任意一名囚徒而言：
- 若同伙选择招供坦白，自身坦白判 5 年，抗拒判 10 年（坦白更优）；
- 若同伙选择攻守同盟抗拒，自身坦白直接无罪释放（0 年），抗拒判 1 年（坦白依然更优！）；
因此，“招供背叛”对于每个人而言都是不可动摇的**绝对严格占优策略**。每个人出于严密的个体自利理性必然选择背叛，最终导致双方均被判 5 年徒刑；而若双方均“非理性”地选择合作抗拒，本可双双只判 1 年。这揭示了个体局部最优与社会全局最优之间的深层内在制度性断裂。

---

### 习题 24.4 (Exercise 24.4)
**题目**：在 Level-$k$ 思考模型中，设一个博弈为“猜均值的 $2/3$”（每个人从 $0$ 到 $100$ 中选择一个整数，猜的数字最接近所有人提交平均值之 $2/3$ 的人获胜）。分析 Level 0、Level 1、Level 2 与无限阶理性的纳什均衡出价。

**详细解答**：
1. **Level 0 智能体**：完全无知，在 $[0, 100]$ 内均匀随机随机选择，平均出价约为 **50**；
2. **Level 1 智能体**：假设全场都是 Level 0（平均出价 50），自身计算最优反应：$50 \times \frac{2}{3} \approx \mathbf{33}$；
3. **Level 2 智能体**：假设全场都是 Level 1（出价 33），自身计算最优反应：$33 \times \frac{2}{3} \approx \mathbf{22}$；
4. **无限阶完全理性（纳什均衡）**：通过无穷轮递归迭代：$x \leftarrow \frac{2}{3} x$，唯一的全局不动点解为全员出价严格等于 **0**！
（实证经济学实验表明，人类初次参与该游戏时的大多数出价集中在 33 与 22 之间，精准印证了 Level 1 与 Level 2 的实证预言）。
