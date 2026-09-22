# 第 19 章：置信状态 (Beliefs)

在前面的所有章节中，我们均假设智能体能够精确且完全地获知环境的真实物理状态 $s$。然而在现实的物理世界中，**传感器的精度永远是受限的、带有随机噪声的，甚至存在物理视距遮挡**。例如水下自主潜航器无法直接得知深海洋流速度、医生无法直接透视患者体内的全部生化病灶、火星车无法直接感知轮胎下松散沙土的深层剪切力。

当真实状态无法被直接访问时，智能体必须将其决策建立在对所有潜在可能状态的统计概率分布之上，这一概率分布被称为**置信状态（Belief State）**。本章系统探讨**部分可观测马尔可夫决策过程（Partially Observable Markov Decision Processes, POMDP）**的形式化框架与置信状态滤波更新理论。我们首先引入 POMDP 的数学七元组；随后推导离散状态空间下的精确贝叶斯滤波更新公式；接着探讨连续线性高斯系统中的**卡尔曼滤波（Kalman Filter）**；随后剖析面向非线性系统的**扩展卡尔曼滤波（EKF）**与**无迹卡尔曼滤波（UKF）**；最后系统阐述适用于任意非线性非高斯环境的非参数化蒙特卡洛利器——**粒子滤波（Particle Filtering）**。

---

## 19.1 POMDP 问题形式化 (POMDP Problem Formulation)

**部分可观测马尔可夫决策过程（POMDP）**通过在标准完全可观测 MDP 的基础上叠加一个带噪声的观测传感器通道构建而成。一个离散时间的平稳 POMDP 由**七元组 $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma)$** 精确形式化：

1. **状态空间 $\mathcal{S}$**：环境隐藏在幕后的真实物理状态集合；
2. **行动空间 $\mathcal{A}$**：智能体可选的决策行动集合；
3. **观测空间 $\mathcal{O}$**：智能体传感器能够接收到的测量读数集合；
4. **状态转移模型 $T(s' \mid s, a)$**：物理世界的真实转移条件概率；
5. **观测模型（Observation Model）$O(o \mid a, s')$**：在执行动作 $a$ 且环境物理转移至新状态 $s'$ 后，传感器生成特定观测值 $o \in \mathcal{O}$ 的条件概率分布（在连续观测下为条件密度函数）；
6. **奖励函数 $R(s, a)$**：依赖于真实底层物理状态与动作的标量即时效用；
7. **折扣因子 $\gamma \in [0, 1)$**：未来贴现衰减因子。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_19_1.png" alt="POMDP 动态决策网络结构" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 19.1：POMDP 的动态决策网络（Dynamic Decision Network）展开图。隐藏状态 $S_t$ 无法直接触达，智能体仅能通过带有噪声的观测节点 $O_t$ 间接感知环境。</p>
</div>

```julia
# POMDP 抽象模型数据结构定义 (来自官方 Julia 算法实现)
struct POMDP
    γ   # discount factor
    𝒮   # state space
    𝒜   # action space
    𝒪   # observation space
    T   # transition function
    R   # reward function
    O   # observation function
    TRO # sample transition, reward, and observation
end
####################
```

---

## 19.2 离散置信状态与贝叶斯滤波更新 (Discrete Belief Updates)

在离散状态空间下，智能体对真实状态的全部认知被浓缩为一个概率分布向量 $\mathbf{b} \in \Delta^{|\mathcal{S}| - 1}$，称为**置信状态（Belief State）**。其中分量 $b(s)$ 表示智能体认为当前真实物理状态正好是 $s$ 的概率，满足：
$$
b(s) \ge 0, \quad \sum_{s \in \mathcal{S}} b(s) = 1
$$
所有合法置信向量构成的空间几何是一个 $|\mathcal{S}| - 1$ 维的标准**概率单纯形（Probability Simplex）**。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_19_2.png" alt="三状态单纯形置信空间" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 19.2：包含 3 个离散物理状态的 POMDP 所对应的二维正三角形置信单纯形空间。三个顶点分别对应对特定单一状态的完全确定性信念。</p>
</div>

### 贝叶斯滤波确切更新公式
设当前置信为 $b$。智能体执行动作 $a$ 并随后观察到传感器读数 $o$。根据贝叶斯定理与马尔可夫链式展开，后验更新的新置信状态 $b'(s')$ 由下式精确给出：
$$
b'(s') = \frac{O(o \mid a, s') \sum_{s \in \mathcal{S}} T(s' \mid s, a) b(s)}{P(o \mid b, a)} = \frac{O(o \mid a, s') \sum_{s \in \mathcal{S}} T(s' \mid s, a) b(s)}{\sum_{s'' \in \mathcal{S}} O(o \mid a, s'') \sum_{s \in \mathcal{S}} T(s'' \mid s, a) b(s)}
$$
记该确定性确定性置信推演映射为算子：$b' = \text{Update}(b, a, o)$。
**信息完备性定理（Aström, 1965）**证明：**置信状态 $b$ 是历史所有观测与行动序列 $(a_0, o_0, \dots, a_{t-1}, o_{t-1})$ 的充分统计量（Sufficient Statistic）！** 智能体完全无需记忆漫长的历史序列，仅凭当前置信向量 $b$ 即可制定全局最优决策。

```julia
# 离散置信状态贝叶斯更新实现
function update(b::Vector{Float64}, 𝒫, a, o)
	𝒮, T, O = 𝒫.𝒮, 𝒫.T, 𝒫.O
    b′ = similar(b)
    for (i′, s′) in enumerate(𝒮)
        po = O(a, s′, o)
        b′[i′] = po * sum(T(s, a, s′) * b[i] for (i, s) in enumerate(𝒮))
    end
    if sum(b′) ≈ 0.0
    	fill!(b′, 1)
    end
    return normalize!(b′, 1)
end
####################
```

---

## 19.3 连续状态与卡尔曼滤波 (Kalman Filter)

对于连续状态与连续动作，最经典的解析置信更新是**卡尔曼滤波（Kalman Filter, Kalman, 1960）**。

### 线性高斯系统假设：
- 状态转移服从线性动力学叠加高斯噪声：$s_{t+1} = \mathbf{F} s_t + \mathbf{B} a_t + \mathbf{w}_t, \; \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_w)$；
- 传感器观测服从线性投影叠加高斯噪声：$o_t = \mathbf{H} s_t + \mathbf{v}_t, \; \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_v)$。

若初始先验置信服从高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$，则在经历任意动作与观测后，**其后验置信状态严格保持为高斯分布**！

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_19_3.png" alt="连续高斯置信变换几何" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 19.3：线性动力学对高斯置信椭圆的拉伸平移预测（左），与融入传感器测量后的后验高斯椭圆方差紧缩（右）。</p>
</div>

### 两步闭式更新：
1. **预测步（Time Update / Predict）**：
   $$
   \boldsymbol{\mu}_{p} = \mathbf{F} \boldsymbol{\mu} + \mathbf{B} a, \quad \mathbf{\Sigma}_{p} = \mathbf{F} \mathbf{\Sigma} \mathbf{F}^\top + \mathbf{\Sigma}_w
   $$
2. **更新步（Measurement Update / Correct）**：
   计算**卡尔曼增益（Kalman Gain）** $\mathbf{K}$：
   $$
   \mathbf{K} = \mathbf{\Sigma}_{p} \mathbf{H}^\top \left( \mathbf{H} \mathbf{\Sigma}_{p} \mathbf{H}^\top + \mathbf{\Sigma}_v \right)^{-1}
   $$
   更新后验均值与后验协方差矩阵：
   $$
   \boldsymbol{\mu}' = \boldsymbol{\mu}_{p} + \mathbf{K} (o - \mathbf{H} \boldsymbol{\mu}_{p}), \quad \mathbf{\Sigma}' = (\mathbf{I} - \mathbf{K} \mathbf{H}) \mathbf{\Sigma}_{p}
   $$

```julia
# 卡尔曼滤波算法实现
struct KalmanFilter
	μb # mean vector
	Σb # covariance matrix
end

function update(b::KalmanFilter, 𝒫, a, o)
	μb, Σb = b.μb, b.Σb
	Ts, Ta, Os = 𝒫.Ts, 𝒫.Ta, 𝒫.Os
	Σs, Σo = 𝒫.Σs, 𝒫.Σo
	# predict
	μp = Ts*μb + Ta*a
	Σp = Ts*Σb*Ts' + Σs
	# update
	Σpo = Σp*Os'
	K = Σpo/(Os*Σp*Os' + Σo)
	μb′ = μp + K*(o - Os*μp)
	Σb′ = (I - K*Os)*Σp
	return KalmanFilter(μb′, Σb′)
end
####################
```

---

## 19.4 扩展与无迹卡尔曼滤波 (EKF & UKF)

当物理系统包含非线性方程时（$s_{t+1} = f(s_t, a_t) + w_t$，$o_t = g(s_t) + v_t$）：
1. **扩展卡尔曼滤波（Extended Kalman Filter, EKF）**：在当前均值点处计算动力学函数的一阶偏导雅可比矩阵（Jacobians）进行局部切线线性化。若非线性强烈，一阶线性化会导致严重的误差发散；
2. **无迹卡尔曼滤波（Unscented Kalman Filter, UKF, Julier & Uhlmann, 1997）**：彻底放弃雅可比矩阵求导，转而利用**无迹变换（Unscented Transform）**在均值周围精选挑选 $2d+1$ 个确定性**Sigma 采样点**，直接将这些点代入真实非线性函数映射，能够精确捕获非线性分布的二阶甚至三阶泰勒展开矩。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_19_4.png" alt="无迹变换 Sigma 点散布" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 19.4：不同尺度缩放参数 $\lambda$ 对无迹变换高维空间中围绕均值点辐射的 Sigma 点空间几何构型的影响。</p>
</div>

---

## 19.5 粒子滤波 (Particle Filtering)

对于具有严重多模态、断崖不连续、复杂几何障碍的通用连续非线性系统，最通用的非参数解法是**粒子滤波（Particle Filtering / Sequential Monte Carlo, SMC）**。

粒子滤波使用一个由 $m$ 个带权随机粒子构成的离散集合来经验逼近任意形状的复杂置信密度：
$$
b(s) \approx \sum_{i=1}^m w^{(i)} \delta(s - s^{(i)})
$$

### 三大迭代循环：
1. **传播（Propagation）**：根据物理转移模型 $T(s' \mid s^{(i)}, a)$ 对每个粒子随机抽样演进，模拟物理扩散；
2. **重加权（Weighting）**：观察到新测量值 $o$ 后，根据测量似然赋予粒子权重：$w^{(i)} \propto O(o \mid a, s^{(i)})$；
3. **重要性重采样（Resampling）**：当有效粒子数衰减时，根据权重高低进行轮盘赌复制高权重粒子、淘汰近乎归零的劣质粒子，彻底化解**粒子退化（Particle Depletion）**难题。

```julia
# 粒子滤波算法实现
struct ExtendedKalmanFilter
	μb # mean vector
	Σb # covariance matrix
end

import ForwardDiff: jacobian
function update(b::ExtendedKalmanFilter, 𝒫, a, o)
	μb, Σb = b.μb, b.Σb
	fT, fO = 𝒫.fT, 𝒫.fO
	Σs, Σo = 𝒫.Σs, 𝒫.Σo
	# predict
	μp = fT(μb, a)
	Ts = jacobian(s->fT(s, a), μb)
	Os = jacobian(fO, μp)
	Σp = Ts*Σb*Ts' + Σs
	# update
	Σpo = Σp*Os'
	K = Σpo/(Os*Σp*Os' + Σo)
	μb′ = μp + K*(o - fO(μp))
	Σb′ = (I - K*Os)*Σp
	return ExtendedKalmanFilter(μb′, Σb′)
end
####################
```

---

## 19.6 本章小结 (Summary)

- **认知与现实的解耦**：POMDP 将底层真实物理状态与智能体认知层面的置信状态分离开来，建立起部分可观测决策的标准形式化；
- **置信状态的完备性**：历史观测动作序列的全部因果信息可以被完全无损压缩至当前置信分布向量，奠定了置信空间规划的理论底座；
- **离散贝叶斯与连续卡尔曼**：离散状态通过概率单纯形贝叶斯更新，线性高斯系统通过卡尔曼增益实现矩阵闭式投影；
- **非线性滤波演进**：从雅可比线性化的 EKF，到确定性采样保持高阶精度的 UKF，再到能够拟合任意多模态密度的非参数粒子滤波。

---

## 19.7 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 19.1 (Exercise 19.1)
**题目**：在经典的“啼哭婴儿”（Crying Baby）POMDP 问题中，婴儿状态有两个：饱腹（$s_0$）与饥饿（$s_1$）。动作集合包含喂食与不作为。若当前置信为婴儿处于饥饿状态的概率 $b(s_1) = 0.5$（此时处于饱腹概率 $b(s_0) = 0.5$）。智能体执行动作“不作为”（$a_0$）。已知在不作为动作下，婴儿如果处于饱腹，有 $10\%$ 的概率会转化为饥饿：$T(s_1 \mid s_0, a_0) = 0.1$；若原本饥饿则必然保持饥饿：$T(s_1 \mid s_1, a_0) = 1.0$。若随后听到婴儿发出了哭声（$o_1$），已知哭声的观测似然为：饥饿时哭泣概率 $O(o_1 \mid a_0, s_1) = 0.8$；饱腹时哭泣概率 $O(o_1 \mid a_0, s_0) = 0.1$。求听见哭声后，婴儿处于饥饿状态的后验置信值 $b'(s_1)$。

**详细解答**：
1. **预测步（物理转移传播）**：
   计算执行不作为动作后、观察到哭声前，婴儿处于饥饿与饱腹的先验概率：
   $$
   \begin{aligned}
   P(s'_1 \mid b, a_0) &= T(s_1 \mid s_0, a_0) b(s_0) + T(s_1 \mid s_1, a_0) b(s_1) = 0.1 \times 0.5 + 1.0 \times 0.5 = 0.05 + 0.50 = 0.55 \\
   P(s'_0 \mid b, a_0) &= 1 - P(s'_1 \mid b, a_0) = 1 - 0.55 = 0.45
   \end{aligned}
   $$
2. **更新步（融入哭声观测似然）**：
   计算未归一化的后验概率分子：
   - 饥饿状态分子：$O(o_1 \mid a_0, s_1) P(s'_1 \mid b, a_0) = 0.8 \times 0.55 = 0.44$；
   - 饱腹状态分子：$O(o_1 \mid a_0, s_0) P(s'_0 \mid b, a_0) = 0.1 \times 0.45 = 0.045$；
3. **归一化常数（分母）**：
   $$
   P(o_1 \mid b, a_0) = 0.44 + 0.045 = 0.485
   $$
4. **计算后验置信**：
   $$
   b'(s_1) = \frac{0.44}{0.485} = \frac{440}{485} = \frac{88}{97} \approx \mathbf{0.9072}
   $$
听见婴儿啼哭后，智能体对婴儿处于饥饿状态的置信度从 $50\%$ 骤增至约 $90.72\%$。

---

### 习题 19.2 (Exercise 19.2)
**题目**：在卡尔曼滤波中，若一维系统的先验不确定性方差为 $\sigma_p^2 = 4.0$，传感器测量噪声方差为 $\sigma_v^2 = 1.0$（测量矩阵 $H = 1$）。求卡尔曼增益 $K$ 与更新后的后验方差 $\sigma'^2$。

**详细解答**：
1. 代入一维卡尔曼增益公式：
   $$
   K = \frac{\sigma_p^2}{\sigma_p^2 + \sigma_v^2} = \frac{4.0}{4.0 + 1.0} = \frac{4.0}{5.0} = \mathbf{0.8}
   $$
2. 代入后验协方差更新公式：
   $$
   \sigma'^2 = (1 - K) \sigma_p^2 = (1 - 0.8) \times 4.0 = 0.2 \times 4.0 = \mathbf{0.8}
   $$
融入传感器测量后，状态方差从 4.0 急剧收缩至 0.8。

---

### 习题 19.3 (Exercise 19.3)
**题目**：在上题的设定中，若传感器测量噪声极其恶劣，方差趋于无穷大（$\sigma_v^2 \to \infty$），卡尔曼增益与后验估计会发生什么现象？

**详细解答**：
当 $\sigma_v^2 \to \infty$ 时：
$$
K = \lim_{\sigma_v^2 \to \infty} \frac{\sigma_p^2}{\sigma_p^2 + \sigma_v^2} = 0
$$
后验均值更新式变为：
$$
\mu' = \mu_p + 0 \cdot (o - \mu_p) = \mu_p
$$
后验方差更新式变为：
$$
\sigma'^2 = (1 - 0) \sigma_p^2 = \sigma_p^2
$$
**物理意义**：当传感器噪声极大、读数完全不可信时，卡尔曼滤波自动将传感器权重置零，完全忽略新的观测数据，纯粹依赖物理动力学模型的预测外推。

---

### 习题 19.4 (Exercise 19.4)
**题目**：在粒子滤波中，什么是“粒子退化”（Particle Depletion）现象？工程中通常采用什么定量指标来触发重采样？

**详细解答**：
- **粒子退化**：随着时间步推移，由于乘法权重的累积衰减，绝大多数粒子的重要性权重会逐渐趋近于零，最终全种群的权重几乎被极少数甚至单个粒子所独占，导致大量算力被浪费在权重为零的僵尸粒子上，粒子云丧失空间多样性；
- **触发指标**：工程中通常通过计算**有效粒子数（Effective Sample Size, ESS / $N_{\text{eff}}$）**：
  $$
  N_{\text{eff}} = \frac{1}{\sum_{i=1}^m (w^{(i)})^2}
  $$
  当 $N_{\text{eff}}$ 跌破预设安全阈值（如 $N_{\text{eff}} < m / 2$）时，立即强制触发重要性重采样。

---

### 习题 19.5 (Exercise 19.5)
**题目**：在无迹卡尔曼滤波（UKF）中，对于一个 $d$ 维连续状态向量，无迹变换最少需要生成多少个确定性 Sigma 采样点？

**详细解答**：
对于 $d$ 维高斯分布，无迹变换在均值点中心放置 1 个点，并在每个特征轴正负方向各放置 1 个关于协方差椭球半径对称的边缘点，总共生成 **$2d + 1$ 个确定性 Sigma 采样点**。
（例如三维刚体姿态估计仅需 $2 \times 3 + 1 = 7$ 个 Sigma 点即可精确捕获二阶非线性统计矩）。
