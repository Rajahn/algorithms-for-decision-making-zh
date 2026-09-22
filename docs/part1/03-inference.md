# 第 3 章：推断 (Inference)

在构建完成贝叶斯网络等概率表示模型后，智能体面临的核心任务是利用该模型进行**概率推断（Probabilistic Inference）**。推断是指在观察到部分变量的取值（即**证据，Evidence**）后，计算网络中其他未观测变量（即**查询变量，Query**）的后验概率分布。

本章系统探讨贝叶斯网络中的各类精确推断与近似推断算法。我们首先介绍朴素贝叶斯（Naive Bayes）模型的解析推断，随后引出适用于通用图结构的**和-积变量消除法（Sum-Product Variable Elimination）**与**信度传播（Belief Propagation）**。在证明一般精确推断属于 NP-hard 之后，我们深入剖析基于蒙特卡洛采样的近似算法，包括**直接采样（Direct Sampling）**、**似然加权采样（Likelihood Weighted Sampling）**与**吉布斯采样（Gibbs Sampling，MCMC）**。最后，我们给出多元高斯模型在连续域下的精确条件推断解析解。


::: tip 🧭 概念进阶之梯：如何直觉理解本章

- **核心心智模型（侦探破案）**：传感器测到的物理读数是“案发现场证据 $\mathbf{E} = \mathbf{e}$”，嫌疑人的作案动机是“隐藏查询变量 $\mathbf{Q}$”。推断任务就是：在锁定证据切片后，将其他所有无关干扰变量 $\mathbf{H}$ 边际化累加消除，计算后验概率 $P(\mathbf{Q} \mid \mathbf{E}=\mathbf{e})$。
- **算法演进阶梯（从算死到巧算，再到抽样）**：
  1. **暴力穷举**：枚举所有隐藏变量可能组合求和——指数级超时；
  2. **和-积变量消除法**：利用乘法分配律 $ab + ac = a(b+c)$，把求和符号推进到局部因子内——但树宽过大时仍是 NP-hard；
  3. **信度传播（BP）**：局部相邻节点互发前向预测与后向诊断消息——树形图上 $O(n)$ 线性极速收敛；
  4. **蒙特卡洛抽样**：算不出解析解就掷骰子模拟！直接采样容易因证据罕见而全盘丢弃，**似然加权**强制固定证据并按似然赋权，**吉布斯采样（MCMC）**仅利用每个变量的局部马尔可夫毯做周期状态轮换。
- **核心数理骨架**：高斯模型的条件推断不需要任何网格枚举，其条件分布依旧是高斯分布，**舒尔补（Schur Complement）**直接给出了条件协方差的矩阵闭式解析解。
:::
---

## 3.1 贝叶斯网络中的推断任务 (Inference in Bayesian Networks)

在一个包含随机变量集合 $\mathbf{X} = \{X_1, \dots, X_n\}$ 的贝叶斯网络中，推断问题通常将网络中的变量划分为三个互不相交的子集：
1. **证据变量集合 $\mathbf{E}$**：其实际物理状态已被传感器观测到，赋值记作 $\mathbf{e}$；
2. **查询变量集合 $\mathbf{Q}$**：我们希望推断其后验概率分布的目标变量；
3. **未观测隐藏变量集合 $\mathbf{H}$**：既未被观测、也非直接查询目标的潜在干扰变量（满足 $\mathbf{X} = \mathbf{Q} \cup \mathbf{E} \cup \mathbf{H}$）。

推断的核心目标是计算条件概率分布 $P(\mathbf{Q} \mid \mathbf{E} = \mathbf{e})$。根据条件概率定义与全概率边际化法则，后验分布可展开为：
$$
P(\mathbf{Q} \mid \mathbf{E} = \mathbf{e}) = \frac{P(\mathbf{Q}, \mathbf{e})}{P(\mathbf{e})} = \frac{\sum_{\mathbf{h}} P(\mathbf{Q}, \mathbf{e}, \mathbf{h})}{\sum_{\mathbf{q}'} \sum_{\mathbf{h}} P(\mathbf{q}', \mathbf{e}, \mathbf{h})} \propto \sum_{\mathbf{h}} P(\mathbf{Q}, \mathbf{e}, \mathbf{h})
$$
式中分母 $P(\mathbf{e})$ 是与查询取值无关的归一化常数。由于全联合概率可以通过贝叶斯网络的局部条件概率乘积计算，暴力推断法需要对所有隐藏变量 $\mathbf{h}$ 的指数级可能组合进行枚举求和。当隐藏变量较多时，暴力枚举完全不可行。

```julia
# 抽象精确推断接口定义 (来自官方 Julia 算法实现)
function Base.:*(ϕ::Factor, ψ::Factor)
    ϕnames = variablenames(ϕ)
    ψnames = variablenames(ψ)
    ψonly = setdiff(ψ.vars, ϕ.vars)
    table = FactorTable()
    for (ϕa,ϕp) in ϕ.table
        for a in assignments(ψonly)
            a = merge(ϕa, a)
            ψa = select(a, ψnames)
            table[a] = ϕp * get(ψ.table, ψa, 0.0)
        end
    end
    vars = vcat(ϕ.vars, ψonly)
    return Factor(vars, table)
end
####################
```

---

## 3.2 朴素贝叶斯模型 (Inference in Naive Bayes Models)

**朴素贝叶斯（Naive Bayes）**是一类结构极其简洁但在工程分类中极具威力的概率图模型。

在朴素贝叶斯模型中，存在一个未观测的目标分类变量 $C \in \{c_1, \dots, c_m\}$，以及一组可观测的属性特征变量 $\mathbf{F} = \{F_1, \dots, F_n\}$。其核心假设是：**在给定类别 $C$ 的条件下，所有特征变量之间互相条件独立**。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_3_2.png" alt="朴素贝叶斯星形拓扑" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 3.1：朴素贝叶斯模型的星形有向图结构。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_3_3.png" alt="板块表示法 Plate Notation" style="max-width: 180px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 3.2：利用板块（Plate）紧凑表示 $n$ 个特征变量。</p>
  </div>
</div>

基于该条件独立性假设，联合概率分布可以因式分解为：
$$
P(C, F_1, \dots, F_n) = P(C) \prod_{i=1}^n P(F_i \mid C)
$$
当观测到特征证据 $\mathbf{f} = (f_1, \dots, f_n)$ 时，类别 $C$ 的后验推断具有极简的闭式解析解，完全避免了隐藏变量求和：
$$
P(C = c \mid F_1 = f_1, \dots, F_n = f_n) \propto P(C = c) \prod_{i=1}^n P(F_i = f_i \mid C = c)
$$
推断算法只需在 $O(n)$ 时间内对各特征的条件概率进行累乘并归一化即可完成。

---

## 3.3 和-积变量消除法 (Sum-Product Variable Elimination)

对于通用拓扑结构的贝叶斯网络，最经典的精确推断算法是**和-积变量消除法（Sum-Product Variable Elimination）**。该算法通过利用分配律（Distributive Law），将求和符号深入到各个局部因子之中，从而避免直接在高维联合空间中求和。

### 3.3.1 核心代数运算：因子乘积与因子边际化
变量消除法将网络中的所有条件概率表统一视为**因子（Factors）**：
1. **因子乘积（Factor Product）**：设因子 $\phi_1(\mathbf{X})$ 与 $\phi_2(\mathbf{Y})$，其乘积因子 $\psi(\mathbf{X} \cup \mathbf{Y}) = \phi_1 \cdot \phi_2$ 满足：
   $$
   \psi(\mathbf{x} \cup \mathbf{y}) = \phi_1(\mathbf{x}) \cdot \phi_2(\mathbf{y})
   $$
2. **因子边际化（Factor Marginalization / Sum-Out）**：从因子 $\phi(\mathbf{X})$ 中消除变量 $Y \in \mathbf{X}$：
   $$
   (\sum_Y \phi)(\mathbf{X} \setminus \{Y\}) = \sum_{y} \phi(\mathbf{X} \setminus \{Y\}, Y = y)
   $$
3. **因子条件化（Conditioning）**：当变量 $E \in \mathbf{X}$ 被观测为常数 $e$ 时，将不符合该观测的条目剔除，仅保留 $E=e$ 的切片因子。

```julia
# 因子乘积与因式边际化算法实现
function marginalize(ϕ::Factor, name)
	table = FactorTable()
	for (a, p) in ϕ.table
		a′ = delete!(copy(a), name)
		table[a′] = get(table, a′, 0.0) + p
	end
	vars = filter(v -> v.name != name, ϕ.vars)
	return Factor(vars, table)
end
####################

# 因子观测条件化算法实现
in_scope(name, ϕ) = any(name == v.name for v in ϕ.vars)

function condition(ϕ::Factor, name, value)
	if !in_scope(name, ϕ)
		return ϕ
	end
	table = FactorTable()
	for (a, p) in ϕ.table
		if a[name] == value
			table[delete!(copy(a), name)] = p
		end
	end
	vars = filter(v -> v.name != name, ϕ.vars)
	return Factor(vars, table)
end

function condition(ϕ::Factor, evidence)
	for (name, value) in pairs(evidence)
		ϕ = condition(ϕ, name, value)
	end
	return ϕ
end
####################
```

### 3.3.2 变量消除算法流程
算法按预设的变量消除序（Elimination Ordering）逐一消除所有隐藏变量 $H_k$：
1. 收集所有作用域中包含 $H_k$ 的因子集合 $\{\phi_{i_1}, \dots, \phi_{i_m}\}$；
2. 计算这些因子的点积：$\psi = \prod_{j} \phi_{i_j}$；
3. 对变量 $H_k$ 执行求和消除：$\tau = \sum_{H_k} \psi$；
4. 从当前因子池中移除旧因子，将新因子 $\tau$ 放入因子池；
5. 重复上述步骤直至所有隐藏变量均被消除，最后将剩余因子点积并归一化，输出后验分布。

```julia
# 和-积变量消除算法核心实现
struct ExactInference end

function infer(M::ExactInference, bn, query, evidence)
	ϕ = prod(bn.factors)
	ϕ = condition(ϕ, evidence)
	for name in setdiff(variablenames(ϕ), query)
		ϕ = marginalize(ϕ, name)
	end
	return normalize!(ϕ)
end
####################
```

::: info 边注 3.1：消除顺序与诱导树宽
变量消除算法的计算复杂度取决于消除过程中产生的**最大因子的变量维度（即诱导树宽，Induced Treewidth）**。寻找全局最优的消除顺序本身是一个 NP-hard 问题，实践中通常采用贪心启发式算法（如每次优先消除相连边最少、或产生新因子维度最小的变量）。
:::

---

## 3.4 信度传播与消息传递 (Belief Propagation)

对于**树形结构（Tree）**或**多聚树结构（Polytree）**的贝叶斯网络，可以采用更为优雅的局部消息传递机制——**信度传播算法（Belief Propagation, BP）**。

在树形网络中，任意两节点之间只有一条唯一路径。每个节点仅需向其相邻节点收发局部统计消息：
- 父节点向子节点发送**前向预测消息** $\pi(x)$；
- 子节点向父节点发送**后向诊断消息** $\lambda(x)$。

在无环树结构上，经过自底向上和自顶向下的两轮消息传递后，所有节点的后验边际概率可在 $O(n)$ 线性时间内精确收敛。对于包含闭环回路的通用网络，学术界提出了**环状信度传播（Loopy Belief Propagation）**，通过迭代传递消息直到局部收敛，该过程等价于求解统计物理中的 Bethe 自由能极小化，能够在绝大多数复杂图上获得极高精度的近似后验。

---

## 3.5 计算复杂度与 NP-Hard 特性 (Computational Complexity)

尽管在特定稀疏网络上推断高效，但**通用贝叶斯网络的精确推断已被严格证明为 NP-hard 问题**（Cooper, 1990）。

这一结论可以通过从著名的 **3-SAT（三合取范式可满足性问题）**多项式时间规约来证明。考虑如下包含 4 个布尔变量与 3 个子句的 3-SAT 实例：
$$
(x_1 \lor \neg x_2 \lor x_3) \land (\neg x_1 \lor x_2 \lor x_4) \land (\neg x_2 \lor \neg x_3 \lor \neg x_4)
$$
我们可以构造如图 3.3 所示的贝叶斯网络：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_3_4.png" alt="3SAT 问题的贝叶斯网络表示" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 3.3：将 3-SAT 问题归约为贝叶斯网络推断任务的拓扑构造。</p>
</div>

1. 每个布尔变量 $x_i$ 对应一个先验均匀分布的二值根节点；
2. 每个子句对应一个布尔确定性节点 $c_j$，其条件概率为其对应三个文字的析取；
3. 设立一个最终合取汇聚节点 $Y$，当且仅当所有子句 $c_j$ 均为真时 $Y=1$。

此时计算边际推断 $P(Y = 1) > 0$ 是否成立，等价于判定该 3-SAT 公式是否存在可满足赋值！由于 3-SAT 是经典的 NP 完全问题，这意味着**不存在能在所有网络拓扑上多项式时间内精确求解推断的算法**。因此，近似采样方法对于大规模工程问题不可或缺。

---

## 3.6 直接采样法与拒绝采样 (Direct Sampling)

**直接采样（Direct Sampling / Forward Ancestral Sampling）**通过顺着贝叶斯网络的有向边拓扑排序方向，从根节点开始依次对其条件概率进行随机抽样，生成网络全局变量的一个完整采样样本。

```julia
# 直接采样算法实现
struct VariableElimination
	ordering # array of variable indices
end

function infer(M::VariableElimination, bn, query, evidence)
	Φ = [condition(ϕ, evidence) for ϕ in bn.factors]
	for i in M.ordering
		name = bn.vars[i].name
		if name ∉ query
			inds = findall(ϕ->in_scope(name, ϕ), Φ)
			if !isempty(inds)
				ϕ = prod(Φ[inds])
				deleteat!(Φ, inds)
				ϕ = marginalize(ϕ, name)
				push!(Φ, ϕ)
			end
		end
	end
	return normalize!(prod(Φ))
end
####################
```

当存在观测证据 $\mathbf{E} = \mathbf{e}$ 时，如果直接采样生成的样本中证据变量取值与真实观测不符，则必须将该样本彻底丢弃，这被称为**拒绝采样（Rejection Sampling）**：
$$
\hat{P}(\mathbf{Q} = \mathbf{q} \mid \mathbf{E} = \mathbf{e}) = \frac{N(\mathbf{Q} = \mathbf{q}, \mathbf{E} = \mathbf{e})}{N(\mathbf{E} = \mathbf{e})}
$$
**致命缺陷**：若证据事件 $\mathbf{E} = \mathbf{e}$ 本身是一个小概率罕见事件（例如在 10 个二值变量中发生特定组合的先验概率可能小于 $10^{-6}$），直接采样的绝大多数样本都会被无情丢弃，采样效率呈指数级衰减。

---

## 3.7 似然加权采样 (Likelihood Weighted Sampling)

为了克服拒绝采样的严重失效，**似然加权采样（Likelihood Weighted Sampling）**应运而生。

其核心思想是：**在采样过程中绝不随机生成证据变量的值，而是强制将所有证据变量固定为其观测值 $\mathbf{e}$**。为了修正人为固定变量带来的概率偏差，每个完整样本被赋予一个非负的**权重（Weight）** $w$。该权重定义为：在采样轨迹中遇到每个证据变量时，该变量在其父节点当前采样状态下的条件概率之连乘积：
$$
w = \prod_{E_i \in \mathbf{E}} P(E_i = e_i \mid \text{Parents}(E_i))
$$

```julia
# 似然加权采样算法实现
function topological_sort(G)
	G = deepcopy(G)
	ordering = []
	parentless = filter(i -> isempty(inneighbors(G, i)), 1:nv(G))
	while !isempty(parentless)
		i = pop!(parentless)
		push!(ordering, i)
		for j in copy(outneighbors(G, i))
			rem_edge!(G, i, j)
			if isempty(inneighbors(G, j))
				push!(parentless, j)
			end
		end
	end
	return ordering
end
####################
```

最终，查询变量的后验概率通过带权加权平均估算得出：
$$
\hat{P}(\mathbf{Q} = \mathbf{q} \mid \mathbf{E} = \mathbf{e}) = \frac{\sum_{k: \mathbf{q}^{(k)} = \mathbf{q}} w^{(k)}}{\sum_{k=1}^m w^{(k)}}
$$
似然加权保证生成的每个样本都与证据绝对吻合，无任何样本浪费，在大中型网络中性能显著优于直接拒绝采样。

---

## 3.8 吉布斯采样 (Gibbs Sampling & MCMC)

当证据变量位于网络底层或叶节点时，似然加权采样仍然无法在根节点的采样阶段受到证据的有效引导。为此，现代统计推断广泛采用基于马尔可夫链蒙特卡洛（MCMC）的**吉布斯采样（Gibbs Sampling）**。

### 3.8.1 算法机理
吉布斯采样维护一个当前全网络所有变量的全局状态配置。固定所有证据变量为 $\mathbf{e}$，其余未观测变量初始化为随机有效值。随后，算法循环遍历每一个非证据变量 $Z_i$，在**固定网络中所有其他变量当前状态的条件下，从其局部全条件分布中重新抽取 $Z_i$ 的新值**：
$$
Z_i \sim P(Z_i \mid \mathbf{X} \setminus \{Z_i\})
$$

根据第 2 章推导的条件独立性，给定一个变量的**马尔可夫毯（Markov Blanket，包含其父节点、子节点及配偶节点）**时，该变量与全网其余节点条件独立！因此局部条件分布具有极其紧凑的局部计算公式：
$$
P(Z_i \mid \mathbf{X} \setminus \{Z_i\}) = P(Z_i \mid \text{MB}(Z_i)) \propto P(Z_i \mid \text{Parents}(Z_i)) \prod_{Y_j \in \text{Children}(Z_i)} P(Y_j \mid \text{Parents}(Y_j))
$$

```julia
# 吉布斯采样算法实现 (基于马尔可夫毯局部更新)
function Base.rand(ϕ::Factor)
	tot, p, w = 0.0, rand(), sum(values(ϕ.table))
	for (a,v) in ϕ.table
		tot += v/w
		if tot >= p
			return a
		end
	end
	return Assignment()
end

function Base.rand(bn::BayesianNetwork)
	a = Assignment()
	for i in topological_sort(bn.graph)
		name, ϕ = bn.vars[i].name, bn.factors[i]
		a[name] = rand(condition(ϕ, a))[name]
	end
	return a
end
####################
```

### 3.8.2 案例对比：化学物泄漏检测网络
考虑下图 3.4 所示的化学物泄漏警报网络，以及图 3.5 对直接采样、似然加权采样与吉布斯采样收敛速度的数值基准测试：

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_3_5.png" alt="化学物检测网络" style="max-width: 220px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 3.4：化学物质泄漏监测贝叶斯网络。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_3_6.png" alt="采样推断收敛速度对比" style="max-width: 480px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 3.5：三种采样算法随着样本数量增加的后验估计收敛曲线对比（真值为红色虚线）。</p>
  </div>
</div>

实验表明，吉布斯采样在经历初始的预热期（burn-in）后，能以最低的方差稳定收敛至真实后验概率。

---

## 3.9 多元高斯模型中的精确推断 (Inference in Gaussian Models)

对于连续随机向量，若其联合分布服从多元高斯分布，则条件后验概率分布存在严格的解析闭式解。

设有联合高斯随机向量 $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，将其划分为查询变量子向量 $\mathbf{X}_A$ 与观测证据子向量 $\mathbf{X}_B$：
$$
\mathbf{X} = \begin{bmatrix} \mathbf{X}_A \\ \mathbf{X}_B \end{bmatrix}, \quad \boldsymbol{\mu} = \begin{bmatrix} \boldsymbol{\mu}_A \\ \boldsymbol{\mu}_B \end{bmatrix}, \quad \boldsymbol{\Sigma} = \begin{bmatrix} \boldsymbol{\Sigma}_{AA} & \boldsymbol{\Sigma}_{AB} \\ \boldsymbol{\Sigma}_{BA} & \boldsymbol{\Sigma}_{BB} \end{bmatrix}
$$
当观测到证据 $\mathbf{X}_B = \mathbf{b}$ 时，条件分布 $p(\mathbf{X}_A \mid \mathbf{X}_B = \mathbf{b})$ **依然严格服从多元高斯分布**：
$$
\mathbf{X}_A \mid \mathbf{X}_B = \mathbf{b} \sim \mathcal{N}(\boldsymbol{\mu}_{A \mid B}, \boldsymbol{\Sigma}_{A \mid B})
$$
其后验条件均值与条件协方差矩阵由**舒尔补（Schur Complement）**精确给出：
$$
\begin{aligned}
\boldsymbol{\mu}_{A \mid B} &= \boldsymbol{\mu}_A + \boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} (\mathbf{b} - \boldsymbol{\mu}_B) \\
\boldsymbol{\Sigma}_{A \mid B} &= \boldsymbol{\Sigma}_{AA} - \boldsymbol{\Sigma}_{AB} \boldsymbol{\Sigma}_{BB}^{-1} \boldsymbol{\Sigma}_{BA}
\end{aligned}
$$
注意：后验协方差矩阵 $\boldsymbol{\Sigma}_{A \mid B}$ 完全独立于实际观测到的具体数值 $\mathbf{b}$，该不依赖性是卡尔曼滤波能够离线预计算误差协方差演化的理论根基。

```julia
# 高斯模型解析条件推断实现
struct DirectSampling
	m # number of samples
end

function infer(M::DirectSampling, bn, query, evidence)
	table = FactorTable()
	for i in 1:(M.m)
		a = rand(bn)
		if all(a[k] == v for (k,v) in pairs(evidence))
			b = select(a, query)
			table[b] = get(table, b, 0) + 1
		end
	end
	vars = filter(v->v.name ∈ query, bn.vars)
	return normalize!(Factor(vars, table))
end
####################
```

---

## 3.10 本章小结 (Summary)

- **推断的形式化**：通过将变量划分为查询变量 $\mathbf{Q}$、证据变量 $\mathbf{E}$ 与隐藏变量 $\mathbf{H}$，推断本质上是在观测切片下的高维求和与重新归一化；
- **朴素贝叶斯的线性优势**：强条件独立性假设消除了隐藏变量，推断复杂度骤降为特征数量的线性级别 $O(n)$；
- **和-积变量消除法**：利用乘法对加法的分配律，通过局部因子乘积与边际化避免在全联合空间枚举，树宽决定了其计算时间上限；
- **复杂度壁垒**：通用贝叶斯网络推断可归约为 3-SAT，属于 NP-hard，这决定了必须发展高效的近似采样体系；
- **采样推断谱系**：直接采样易因证据罕见而遭受指数级样本丢弃；似然加权强制固定证据并通过似然赋予权重；吉布斯采样（MCMC）利用马尔可夫毯的局部全条件分布进行周期迭代，显著降低估算方差；
- **高斯共轭美感**：多元高斯联合分布在观测证据切片下保持高斯闭包性，舒尔补公式提供了直接的矩阵代数解析解。

---

## 3.11 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 3.1 (Exercise 3.1)
**题目**：考虑二值变量贝叶斯网络 $A \to B \to C$。写出求解后验查询 $P(a^1 \mid c^1)$ 的精确推断公式。

**详细解答**：
根据贝叶斯定理与边际化全概率公式：
$$
P(a^1 \mid c^1) = \frac{P(a^1, c^1)}{P(c^1)} = \frac{\sum_{b} P(a^1, b, c^1)}{\sum_{a} \sum_{b} P(a, b, c^1)}
$$
代入网络图的因式分解定理 $P(A, B, C) = P(A)P(B \mid A)P(C \mid B)$：
$$
P(a^1 \mid c^1) = \frac{P(a^1) \sum_{b} P(b \mid a^1) P(c^1 \mid b)}{\sum_{a} P(a) \sum_{b} P(b \mid a) P(c^1 \mid b)}
$$
展开对二值隐藏变量 $B \in \{b^0, b^1\}$ 的求和：
分子为 $P(a^1) \left[ P(b^0 \mid a^1)P(c^1 \mid b^0) + P(b^1 \mid a^1)P(c^1 \mid b^1) \right]$，分母为对 $A \in \{a^0, a^1\}$ 重复该求和计算并累加。

---

### 习题 3.2 (Exercise 3.2)
**题目**：对于卫星网络拓扑（$B \to E \leftarrow S$，$E \to D$，$E \to C$），写出求解后验 $P(b^1 \mid d^1)$ 的和-积推断表达式，说明如何安排求和顺序以减少计算量。

**详细解答**：
联合分布因式分解为 $P(B, S, E, D, C) = P(B)P(S)P(E \mid B, S)P(D \mid E)P(C \mid E)$。
查询目标为 $P(b^1 \mid d^1) \propto \sum_{S} \sum_{E} \sum_{C} P(b^1, S, E, d^1, C)$。
1. 首先观察到变量 $C$ 仅出现在项 $P(C \mid E)$ 中，对其消除：$\sum_C P(C \mid E) = 1$ 恒成立，因此未受观测且无证据后代的变量 $C$ 可直接从网络中剔除！
2. 剩余公式简化为：
   $$
   P(b^1 \mid d^1) \propto P(b^1) \sum_E P(d^1 \mid E) \sum_S P(S) P(E \mid b^1, S)
   $$
通过将求和深入，先对 $S$ 求和得到因子 $\tau(E)$，再与 $P(d^1 \mid E)$ 相乘后对 $E$ 求和，避免了多重变量联合枚举。

---

### 习题 3.3 (Exercise 3.3)
**题目**：假设自动驾驶感知系统报告前方障碍物尺寸 $S \in \{\text{small}, \text{large}\}$。类别变量 $C \in \{\text{pedestrian}, \text{car}\}$ 的先验为 $P(C=\text{car}) = 0.8$，$P(C=\text{pedestrian}) = 0.2$。已知条件似然 $P(S=\text{small} \mid \text{pedestrian}) = 0.9$，$P(S=\text{small} \mid \text{car}) = 0.1$。若观测到障碍物为 small，求其为行人的后验概率。

**详细解答**：
根据一维贝叶斯法则：
$$
P(C=\text{ped} \mid S=\text{small}) = \frac{P(S=\text{small} \mid \text{ped}) P(\text{ped})}{P(S=\text{small} \mid \text{ped}) P(\text{ped}) + P(S=\text{small} \mid \text{car}) P(\text{car})}
$$
代入数值：
- 分子：$0.9 \times 0.2 = 0.18$；
- 分母：$0.18 + (0.1 \times 0.8) = 0.18 + 0.08 = 0.26$；
计算后验：
$$
P(C=\text{ped} \mid S=\text{small}) = \frac{0.18}{0.26} = \frac{9}{13} \approx \mathbf{0.6923}
$$
尽管行人的先验仅为 0.2，但尺寸为 small 的观测证据使其后验概率大幅提升至约 69.23%。

---

### 习题 3.4 (Exercise 3.4)
**题目**：在图 3.3 所示的 3-SAT 归约网络中，给定布尔子句 $c_3 = (x_2 \lor \neg x_3 \lor x_4)$，求条件概率 $P(c_3=1 \mid x_2=1, x_3=0, x_4=1)$ 的取值。

**详细解答**：
由于 $c_3$ 是确定性布尔逻辑节点，其真值由逻辑表达式决定。
当 $x_2=1$ 时，文字 $x_2$ 为真，因此整个析取式 $(x_2 \lor \neg x_3 \lor x_4)$ 必定为真。
因此其对应的条件概率为：
$$
P(c_3 = 1 \mid x_2=1, x_3=0, x_4=1) = \mathbf{1.0}
$$

---

### 习题 3.5 (Exercise 3.5)
**题目**：给定有向无环图结构，说明拓扑排序（Topological Sort）在直接祖先采样中的关键作用。

**详细解答**：
直接采样算法必须先生成父节点的取值，然后才能基于该父节点的取值去索引条件概率表 $P(X_i \mid \text{Parents}(X_i))$ 以抽取子节点。
**拓扑排序**保证对于图中的每一条有向边 $X_j \to X_i$，父节点 $X_j$ 在排序序列中必然严格出现在子节点 $X_i$ 之前。因此，只要严格按照拓扑排序序列推进，在采样任意节点 $X_i$ 时，其所有父节点的状态值必定已经全部确定，保证了采样算法的顺利推进。

---

### 习题 3.6 (Exercise 3.6)
**题目**：在似然加权采样中，假设网络包含节点 $A \to B \to C$。证据为 $B=1$。若采样器在根节点抽得 $A=0$，且已知条件概率表条目 $P(B=1 \mid A=0) = 0.05$，$P(B=1 \mid A=1) = 0.8$。求该单条采样轨迹的权重增量。

**详细解答**：
在似然加权采样中，非证据节点（如 $A$）正常采样，不产生权重乘数（权重贡献为 1）；
当推进到证据节点 $B$ 时，不进行随机抽取，直接强制赋予观测值 $B=1$，并将其对应的条件似然值累乘到当前样本权重 $w$ 中：
$$
w = w \cdot P(B = 1 \mid A = 0) = 1.0 \times 0.05 = \mathbf{0.05}
$$
该权重客观反映了在 $A=0$ 的假设下生成证据 $B=1$ 的相对似然度。

---

### 习题 3.7 (Exercise 3.7)
**题目**：设学生考试成绩包含数学 $M$、阅读 $R$ 和写作 $W$ 三门科目，服从三元联合高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其中：
$$
\boldsymbol{\mu} = \begin{bmatrix} 70 \\ 75 \\ 72 \end{bmatrix}, \quad \boldsymbol{\Sigma} = \begin{bmatrix} 100 & 50 & 40 \\ 50 & 100 & 60 \\ 40 & 60 & 100 \end{bmatrix}
$$
若某学生数学成绩已测定为 $M = 90$，利用舒尔补公式求该学生阅读成绩 $R$ 的后验条件期望。

**详细解答**：
将变量分为查询变量 $A=\{R\}$ 与证据变量 $B=\{M\}$：
- $\mu_R = 75$，$\mu_M = 70$；
- 协方差项：$\Sigma_{RM} = 50$，$\Sigma_{MM} = 100$；
代入高斯条件期望更新公式：
$$
\mathbb{E}[R \mid M = 90] = \mu_R + \Sigma_{RM} \Sigma_{MM}^{-1} (90 - \mu_M) = 75 + \frac{50}{100} (90 - 70) = 75 + 0.5 \times 20 = 75 + 10 = \mathbf{85}
$$
观测到优秀的数学成绩使该学生的阅读成绩期望提升了 10 分。
