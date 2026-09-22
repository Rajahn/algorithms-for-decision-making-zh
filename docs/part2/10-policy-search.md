# 第 10 章：策略搜索 (Policy Search)

在前面的章节中，我们主要通过估计价值函数（如状态效用 $U(s)$ 或动作价值 $Q(s, a)$），进而通过贪心最大化间接推导出动作策略。这种基于价值的方法虽然经典，但在高维连续动作空间中，单步求解 $\arg\max_{a} Q(s, a)$ 本身就是一个极其耗时的非线性优化难题；此外，价值函数的微小估计扰动可能导致贪心动作发生剧烈的非连续阶跃跳变。

**策略搜索（Policy Search）**代表了一种更为直接的求解范式：直接对策略本身进行参数化建模 $\pi_{\boldsymbol{\theta}}(s)$，并将序贯决策问题直接转化为在参数空间 $\boldsymbol{\theta} \in \mathbb{R}^k$ 中寻找最大化期望累积贴现回报的目标优化问题：
$$
\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} U(\boldsymbol{\theta}) = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}}\left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]
$$

本章系统探讨**无需导数信息（Derivative-Free / Black-Box）**的策略搜索算法体系。我们首先介绍蒙特卡洛策略评估及其深度与样本量折衡；随后探讨确定性模式搜索（Hooke-Jeeves 算法）；接着引出**遗传算法（Genetic Algorithms）**与**交叉熵方法（Cross-Entropy Method, CEM）**；最后深入分析现代黑盒优化的基石——**协方差自适应进化策略（Evolution Strategies, ES）**及其镜像采样方差缩减技术。

---

## 10.1 近似策略评估 (Approximate Policy Evaluation)

在策略搜索中，评估某个特定参数配置 $\boldsymbol{\theta}$ 的性能标量 $U(\boldsymbol{\theta})$，依赖于**蒙特卡洛仿真推演（Monte Carlo Rollouts）**。

从初始状态分布 $s_0 \sim p(s_0)$ 采样出发，让策略 $\pi_{\boldsymbol{\theta}}$ 与环境模型进行多步交互，生成一条长为 $d$ 的状态-动作轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_d)$。重复采集 $m$ 条独立轨迹，策略的经验期望效用为：
$$
\hat{U}(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^{d-1} \gamma^t R(s_t^{(i)}, a_t^{(i)})
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_10_1.png" alt="轨迹采样评估效用示意" style="max-width: 270px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.1：通过从初始分布采样多条轨迹计算策略的经验期望效用。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_10_2.png" alt="轨迹深度与样本数量的影响" style="max-width: 270px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.2：仿真推演深度与采样样本数量对评估方差的收敛影响。</p>
  </div>
</div>

```julia
# 蒙特卡洛策略评估算法实现 (来自官方 Julia 算法实现)
struct MonteCarloPolicyEvaluation
    𝒫 # problem
    b # initial state distribution
    d # depth
    m # number of samples
end

function (U::MonteCarloPolicyEvaluation)(π)
    R(π) = rollout(U.𝒫, rand(U.b), π, U.d)
    return mean(R(π) for i = 1:U.m)
end

(U::MonteCarloPolicyEvaluation)(π, θ) = U(s->π(θ, s))
####################
```

---

## 10.2 局部搜索与 Hooke-Jeeves 模式搜索 (Hooke-Jeeves Method)

当参数维度较低时，可以直接采用确定性的无导数模式搜索算法。**Hooke-Jeeves 算法（Hooke & Jeeves, 1961）**通过在当前参数点周围沿各坐标正交基方向进行探索性步进：
1. 沿每个坐标轴正负方向分别试探步长 $\alpha$；
2. 若某方向的评估收益有所提升，则立即移动至新坐标；
3. 若所有正交方向均无提升，则将步长衰减为 $\alpha \leftarrow \alpha \cdot \beta$（$\beta \in (0, 1)$）；
4. 当步长小于预设容差阈值时终止。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_10_3.png" alt="Hooke-Jeeves 算法在调节器问题中的轨迹" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.3：Hooke-Jeeves 模式搜索算法在二维线性调节器策略优化中的参数爬山轨迹。</p>
</div>

```julia
# Hooke-Jeeves 局部策略搜索实现
struct HookeJeevesPolicySearch
    θ # initial parameterization
    α # step size
    c # step size reduction factor
    ϵ # termination step size
end

function optimize(M::HookeJeevesPolicySearch, π, U)
    θ, θ′, α, c, ϵ = copy(M.θ), similar(M.θ), M.α, M.c, M.ϵ
    u, n = U(π, θ), length(θ)
    while α > ϵ
        copyto!(θ′, θ)
        best = (i=0, sgn=0, u=u)
        for i in 1:n
            for sgn in (-1,1)
                θ′[i] = θ[i] + sgn*α
                u′ = U(π, θ′)
                if u′ > best.u
                    best = (i=i, sgn=sgn, u=u′)
                end
            end
            θ′[i] = θ[i]
        end
        if best.i != 0
            θ[best.i] += best.sgn*α
            u = best.u
        else
            α *= c
        end
    end
    return θ
end
####################
```

---

## 10.3 遗传算法 (Genetic Algorithms)

**遗传算法（Genetic Algorithms, GA）**借鉴达尔文生物进化的“适者生存”原理，维护一个由 $m$ 个候选策略参数构成的**种群（Population）**：
1. **适应度评估**：对种群中每个个体计算其经验效用 $U(\boldsymbol{\theta}_i)$；
2. **选择（Selection）**：根据适应度高低通过轮盘赌或锦标赛机制挑选优质亲代；
3. **交叉（Crossover）**：随机配对亲代个体，交叉融合其参数染色体；
4. **变异（Mutation）**：以一定突变概率向参数添加微小的随机扰动 $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \boldsymbol{\epsilon}$，保持种群多样性。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_10_4.png" alt="遗传策略搜索种群演化" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.4：遗传算法随迭代代数（Generations）演进，种群粒子逐步由分散收敛聚拢至最优参数孤岛。</p>
</div>

```julia
# 遗传策略搜索算法实现
struct GeneticPolicySearch
    θs      # initial population
    σ       # initial standard deviation
    m_elite # number of elite samples
    k_max   # number of iterations
end

function optimize(M::GeneticPolicySearch, π, U)
    θs, σ = M.θs, M.σ
    n, m = length(first(θs)), length(θs)
    for k in 1:M.k_max
        us = [U(π, θ) for θ in θs]
        sp = sortperm(us, rev=true)
        θ_best = θs[sp[1]]
        rand_elite() = θs[sp[rand(1:M.m_elite)]]
        θs = [rand_elite() + σ.*randn(n) for i in 1:(m-1)]
        push!(θs, θ_best)
    end
    return last(θs)
end
####################
```

---

## 10.4 交叉熵方法 (Cross-Entropy Method)

**交叉熵方法（Cross-Entropy Method, CEM, Rubinstein, 1999）**是一种基于参数化概率分布采样的强大全局优化算法。

算法不直接维护离散个体，而是维护一个定义在参数空间上的**高斯搜索分布 $p(\boldsymbol{\theta}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$**：
1. 从当前搜索分布中独立抽取 $m$ 个候选参数向量样本；
2. 评估所有样本的累积回报，挑选出表现最好的前 $k$ 个**精英样本（Elite Samples）**；
3. 调整搜索分布的参数（均值 $\boldsymbol{\mu}$ 与协方差 $\boldsymbol{\Sigma}$），使其在交叉熵（KL 散度）意义下最大程度拟合这批精英样本：
   $$
   \boldsymbol{\mu}_{\text{new}} = \frac{1}{k} \sum_{i=1}^k \boldsymbol{\theta}_{\text{elite}}^{(i)}, \quad \boldsymbol{\Sigma}_{\text{new}} = \frac{1}{k} \sum_{i=1}^k (\boldsymbol{\theta}_{\text{elite}}^{(i)} - \boldsymbol{\mu}_{\text{new}})(\boldsymbol{\theta}_{\text{elite}}^{(i)} - \boldsymbol{\mu}_{\text{new}})^\top
   $$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_10_5.png" alt="交叉熵方法搜索演进" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.5：交叉熵方法在简单调节器任务中高斯采样椭圆的收缩演变过程。</p>
</div>

```julia
# 交叉熵策略搜索算法实现
struct CrossEntropyPolicySearch
    p       # initial distribution
    m       # number of samples
    m_elite # number of elite samples
    k_max   # number of iterations
end

function optimize_dist(M::CrossEntropyPolicySearch, π, U)
    p, m, m_elite, k_max = M.p, M.m, M.m_elite, M.k_max
    for k in 1:k_max
        θs = rand(p, m)
        us = [U(π, θs[:,i]) for i in 1:m]
        θ_elite = θs[:,sortperm(us)[(m-m_elite+1):m]]
        p = Distributions.fit(typeof(p), θ_elite)
    end
    return p
end

function optimize(M, π, U)
    return Distributions.mode(optimize_dist(M, π, U))
end
####################
```

---

## 10.5 进化策略与镜像采样 (Evolution Strategies & Mirrored Sampling)

在现代高维强化学习中，**自然进化策略（Natural Evolution Strategies, NES）**已成为一种能够与复杂梯度强化学习相媲美的大规模分布式求解范式（Salimans et al., OpenAI 2017）。

### 10.5.1 各向同性进化策略
考虑均值为 $\boldsymbol{\theta}$、方差为 $\sigma^2 \mathbf{I}$ 的各向同性高斯搜索分布。我们希望最大化参数扰动下的期望目标：
$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[ U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) ]
$$
利用对数导数恒等式，关于中心参数 $\boldsymbol{\theta}$ 的梯度可直接解析表达为：
$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{\sigma} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}[ U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) \cdot \boldsymbol{\epsilon} ]
$$
这揭示了一个深刻本质：**无需系统物理导数，仅通过在当前参数点周围施加白噪声扰动，并按照扰动所得的效用回报作为权重对扰动方向进行加权平均，即构成了目标函数的无偏蒙特卡洛梯度估计！**

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_10_6.png" alt="秩权重变换曲线" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.6：对样本回报进行秩变换归一化的权重函数曲线。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_10_7.png" alt="进化策略优化轨迹" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.7：进化策略在非凸参数地形中的平滑优化迭代轨迹。</p>
  </div>
</div>

### 10.5.2 镜像采样 (Mirrored Sampling / Antithetic Sampling)
为了大幅降低有限扰动样本带来的梯度方差，工程中普遍采用**镜像对抗采样**：每当抽取一个随机噪声向量 $\boldsymbol{\epsilon}$ 时，同时对称构造一对互为相反数的扰动点：
$$
\boldsymbol{\theta}_+ = \boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}, \quad \boldsymbol{\theta}_- = \boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}
$$
梯度估计直接利用两者的对称中心差分计算：
$$
\hat{\mathbf{g}} = \frac{1}{2 m \sigma} \sum_{i=1}^m \left[ U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}_i) - U(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}_i) \right] \boldsymbol{\epsilon}_i
$$

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_10_8.png" alt="镜像采样方差缩减效果对比" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 10.8：标准独立采样与镜像对称采样在梯度假说估计方差上的显著收敛优势对比。</p>
</div>

```julia
# 进化策略核心算法实现 (支持镜像对抗采样与协方差自适应)
struct EvolutionStrategies
    D       # distribution constructor
    ψ       # initial distribution parameterization
    ∇logp   # log search likelihood gradient
    m       # number of samples
    α       # step factor
    k_max   # number of iterations
end

function evolution_strategy_weights(m)
    ws = [max(0, log(m/2+1) - log(i)) for i in 1:m]
    ws ./= sum(ws)
    ws .-= 1/m
    return ws
end

function optimize_dist(M::EvolutionStrategies, π, U)
    D, ψ, m, ∇logp, α = M.D, M.ψ, M.m, M.∇logp, M.α
    ws = evolution_strategy_weights(m)
    for k in 1:M.k_max
        θs = rand(D(ψ), m)
        us = [U(π, θs[:,i]) for i in 1:m]
        sp = sortperm(us, rev=true)
        ∇ = sum(w.*∇logp(ψ, θs[:,i]) for (w,i) in zip(ws,sp))
        ψ += α.*∇
    end
    return D(ψ)
end
####################
```

---

## 10.6 本章小结 (Summary)

- **直接求解的优越性**：策略搜索避开了连续动作空间贪心求解 $\arg\max_a Q(s, a)$ 的计算陷阱，直接将序贯决策转化为黑盒参数优化；
- **无导数搜索谱系**：从确定性的 Hooke-Jeeves 坐标模式搜索，到基于种群杂交变异的遗传算法，再到利用精英样本自适应缩放高斯协方差的交叉熵方法；
- **进化策略的伪梯度本质**：通过对高斯随机扰动施加效用加权，进化策略以黑盒仿真为底座构造了对参数梯度的无偏估计；
- **镜像采样的方差压制**：成对对称抽样从根本上消除了偶数阶泰勒展开偏差，大幅提升了黑盒参数迭代的数值稳定性。

---

## 10.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 10.1 (Exercise 10.1)
**题目**：考虑利用蒙特卡洛仿真评估策略效用。若每次推演的最大步长截断为 $d$，折扣因子为 $\gamma \in (0, 1)$，单步即时奖励有界 $|R(s, a)| \le R_{\max}$。求因有限截断深度 $d$ 带来的最大理论效用截断误差界。

**详细解答**：
从时间步 $d$ 到无穷时域，未被计入的剩余贴现奖励总和为：
$$
|\text{Error}| = \left| \sum_{t=d}^\infty \gamma^t R(s_t, a_t) \right| \le \sum_{t=d}^\infty \gamma^t |R(s_t, a_t)| \le R_{\max} \sum_{t=d}^\infty \gamma^t
$$
提取公因式 $\gamma^d$：
$$
\sum_{t=d}^\infty \gamma^t = \gamma^d \sum_{k=0}^\infty \gamma^k = \frac{\gamma^d}{1 - \gamma}
$$
故最大理论误差界为：
$$
|\text{Error}| \le \mathbf{\frac{\gamma^d R_{\max}}{1 - \gamma}}
$$
随着截断深度 $d$ 增大，截断误差以几何速率 $\gamma^d$ 指数级衰减。

---

### 习题 10.2 (Exercise 10.2)
**题目**：在遗传算法中，若种群包含 10 个个体，其适应度评估值分别为 $1, 2, 3, 4, 5, 6, 7, 8, 9, 10$。若采用轮盘赌比例选择机制，第 10 个个体（适应度为 10）被选中的概率是多少？

**详细解答**：
计算全种群适应度总和：
$$
\sum_{i=1}^{10} f_i = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = \frac{10 \times 11}{2} = 55
$$
轮盘赌选中概率等于其个体适应度占总适应度的比例：
$$
P(\text{选择个体 10}) = \frac{10}{55} = \frac{2}{11} \approx \mathbf{0.1818}
$$

---

### 习题 10.3 (Exercise 10.3)
**题目**：在交叉熵方法中，若从均值为 $\boldsymbol{\mu}$ 的二维高斯分布中抽取 4 个样本点：$(1, 2), (2, 3), (3, 4), (4, 5)$。若精英样本比例设定为前 $50\%$，求更新后新高斯分布的均值向量 $\boldsymbol{\mu}_{\text{new}}$。

**详细解答**：
样本总数为 $m=4$，前 $50\%$ 精英样本包含 $k = 4 \times 0.5 = 2$ 个最高回报样本。
按数值优劣排序，最后两个样本 $(3, 4)$ 与 $(4, 5)$ 为精英样本。
计算精英样本的算术均值：
$$
\boldsymbol{\mu}_{\text{new}} = \frac{1}{2} \left( \begin{bmatrix} 3 \\ 4 \end{bmatrix} + \begin{bmatrix} 4 \\ 5 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} 7 \\ 9 \end{bmatrix} = \mathbf{\begin{bmatrix} 3.5 \\ 4.5 \end{bmatrix}}
$$

---

### 习题 10.4 (Exercise 10.4)
**题目**：证明镜像采样梯度估计量 $\hat{\mathbf{g}} = \frac{1}{2 \sigma} [U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) - U(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon})] \boldsymbol{\epsilon}$ 是中心梯度的一阶泰勒无偏估计。

**详细解答**：
设效用函数 $U(\cdot)$ 在 $\boldsymbol{\theta}$ 处具有二次连续可微泰勒展开：
$$
\begin{aligned}
U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) &= U(\boldsymbol{\theta}) + \sigma \nabla U(\boldsymbol{\theta})^\top \boldsymbol{\epsilon} + \frac{1}{2} \sigma^2 \boldsymbol{\epsilon}^\top \nabla^2 U(\boldsymbol{\theta}) \boldsymbol{\epsilon} + O(\sigma^3) \\
U(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}) &= U(\boldsymbol{\theta}) - \sigma \nabla U(\boldsymbol{\theta})^\top \boldsymbol{\epsilon} + \frac{1}{2} \sigma^2 \boldsymbol{\epsilon}^\top \nabla^2 U(\boldsymbol{\theta}) \boldsymbol{\epsilon} + O(\sigma^3)
\end{aligned}
$$
两式相减，偶数阶二阶导数项严格对称相消：
$$
U(\boldsymbol{\theta} + \sigma \boldsymbol{\epsilon}) - U(\boldsymbol{\theta} - \sigma \boldsymbol{\epsilon}) = 2 \sigma \nabla U(\boldsymbol{\theta})^\top \boldsymbol{\epsilon} + O(\sigma^3)
$$
代入估计量并关于各向同性高斯白噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 求数学期望（已知 $\mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\epsilon}^\top] = \mathbf{I}$）：
$$
\mathbb{E}[\hat{\mathbf{g}}] = \frac{1}{2\sigma} \mathbb{E}\left[ \left( 2\sigma \boldsymbol{\epsilon}^\top \nabla U(\boldsymbol{\theta}) + O(\sigma^3) \right) \boldsymbol{\epsilon} \right] = \mathbb{E}[\boldsymbol{\epsilon} \boldsymbol{\epsilon}^\top] \nabla U(\boldsymbol{\theta}) + O(\sigma^2) = \nabla U(\boldsymbol{\theta}) + O(\sigma^2)
$$
证毕：当 $\sigma \to 0$ 时该估计量严格无偏，且由于二阶导数项完全消除，其方差显著小于单向独立采样。

---

### 习题 10.5 (Exercise 10.5)
**题目**：在进化策略中，为什么通常对样本回报 $U_1, \dots, U_m$ 进行“秩变换”（Rank Transformation）归一化，而不是直接使用原始数值？

**详细解答**：
1. **抵御极端异常值破坏**：物理仿真中偶发的极端奖励或惩罚爆炸会导致梯度估计被单一样本严重主导绑架；
2. **标度不变性（Scale Invariance）**：秩变换使算法只依赖于候选策略之间的相对优劣顺序，对奖励函数的任意单调递增非线性变换保持完全不变，免除了超参数学习率对特定任务奖励尺度的繁琐调参依赖。

---

### 习题 10.6 (Exercise 10.6)
**题目**：当参数维度达到数万维（例如深度神经网络的权重）时，为什么无导数黑盒策略搜索往往不如基于梯度的策略梯度算法高效？

**详细解答**：
黑盒进化策略在 $k$ 维参数空间中，需要向各个空间正交方向发射足够的随机扰动才能准确重构出高维梯度的主要分量，其所需的**采样样本复杂度通常随参数维度 $k$ 线性甚至二次方增长**。而反向传播（Backpropagation）与解析策略梯度定理能够以几乎常数倍的前向开销直接求出数千万维参数的精确导数方向。

---

### 习题 10.7 (Exercise 10.7)
**题目**：在什么特定硬件架构与应用场景下，进化策略（ES）反而能够击败反向传播策略梯度？

**详细解答**：
1. **大规模分布式并行集群**：ES 每次迭代仅需在成千上万个 CPU 节点上并行独立跑仿真推演，节点间通信仅需传输标量适应度值与微小的随机数种子，通信开销近乎为零，具备近乎无限线性的超级扩展能力；
2. **非平滑不可导环境**：环境包含复杂的接触碰撞、硬开关逻辑或强延迟时，物理导数不存在或充斥着病态局部极值，ES 天然展现出强大的长程跳跃平滑能力。
