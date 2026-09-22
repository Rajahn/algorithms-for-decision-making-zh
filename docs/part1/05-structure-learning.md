# 第 5 章：结构学习 (Structure Learning)

在前两章中，我们探讨了在给定网络有向无环图（DAG）拓扑结构的前提下进行推断与参数学习。然而，因果网络本身的**图结构（Graph Structure）**往往同样是未知的。我们如何直接从观测数据中自动发现变量之间的因果依赖拓扑？

本章系统探讨贝叶斯网络的**结构学习（Structure Learning）**理论与搜索算法。我们首先形式化建立用于评估候选图拓扑优劣的**贝叶斯网络评分准则（Bayesian Scoring Criterion）**与 BIC/MDL 惩罚机制，解释其如何内生实现奥卡姆剃刀原理；随后介绍基于贪心爬山法与先验节点排序的 **K2 启发式搜索算法**；接着探讨具有相同条件独立性断言的**马尔可夫等价类（Markov Equivalence Classes）**与本质图（CPDAG）表示；最后介绍直接在等价类空间中高效巡检的**贪心等价搜索（Greedy Equivalence Search, GES）**。

---

## 5.1 贝叶斯网络评分准则 (Bayesian Network Scoring)

结构学习的一种主要范式是**基于评分的搜索（Score-Based Learning）**：定义一个量化图结构与数据集匹配程度的评分函数，随后在所有可能的 DAG 空间中寻找评分最高的图。

### 5.1.1 为什么极大似然无法用于结构学习？
直觉上，我们可能会尝试寻找使数据极大似然达到最大的图结构：
$$
\hat{G} = \arg\max_{G} \ell(\hat{\boldsymbol{\theta}}_G \mid \mathcal{D})
$$
**致命缺陷**：向网络中添加任意一条有向边，极大似然值必然单调非减！如果仅追求最大化训练数据似然度，算法最终必然会输出一个**全连接的有向无环图（Fully Connected DAG）**，因为全连接图拥有最多的自由参数，能完美拟合训练集中的任何偶然噪声，导致灾难性的**过拟合（Overfitting）**。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_5_2.png" alt="三种连接程度的拓扑结构" style="max-width: 580px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 5.1：真实模型（中）、完全连接模型（左）与完全无连接模型（右）。完全连接模型必然导致严重的统计过拟合。</p>
</div>

### 5.1.2 贝叶斯评分 (Bayesian Score)
贝叶斯方法通过对图模型的所有可能参数进行积分边际化，评估图结构本身的后验概率：
$$
P(G \mid \mathcal{D}) \propto P(G) P(\mathcal{D} \mid G) = P(G) \int P(\mathcal{D} \mid \boldsymbol{\theta}, G) p(\boldsymbol{\theta} \mid G)\, d\boldsymbol{\theta}
$$
在无信息先验下（$P(G)$ 为常数），核心在于计算**边际似然（Marginal Likelihood）** $P(\mathcal{D} \mid G)$。

对于离散变量，在狄利克雷先验参数 $\alpha_{ijk}$ 假设下，边际似然具有解析积分形式（即著名的 **BDeu / Cooper-Herskovits 评分**）：
$$
\text{BayesianScore}(G \mid \mathcal{D}) = \sum_{i=1}^n \sum_{j} \left[ \log \frac{\Gamma(\alpha_{ij0})}{\Gamma(\alpha_{ij0} + N_{ij})} + \sum_{k} \log \frac{\Gamma(\alpha_{ijk} + N_{ijk})}{\Gamma(\alpha_{ijk})} \right]
$$
式中 $\alpha_{ij0} = \sum_k \alpha_{ijk}$，$N_{ij} = \sum_k N_{ijk}$，$\Gamma(\cdot)$ 为伽马函数。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_5_1.png" alt="双节点贝叶斯网络" style="max-width: 180px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 5.2：用于演示评分机制的双节点网络。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_5_3.png" alt="贝叶斯评分随数据规模演化" style="max-width: 300px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 5.3：随着数据集样本量 $m$ 增长，真实模型的贝叶斯评分（红色实线）显著超越全连接模型与无连接模型。</p>
  </div>
</div>

贝叶斯评分天生内置了**奥卡姆剃刀惩罚**：参数维度更高的复杂模型由于其先验概率质量被稀疏分散在更高维的空间中，其边际似然积分会自动受到严厉惩罚。

### 5.1.3 BIC / MDL 准则
当样本量 $m \to \infty$ 时，利用拉普拉斯积分渐近展开，贝叶斯评分在大样本极限下严格收敛至**贝叶斯信息准则（Bayesian Information Criterion, BIC）**：
$$
\text{BIC}(G \mid \mathcal{D}) = \ell(\hat{\boldsymbol{\theta}}_G \mid \mathcal{D}) - \frac{\text{dim}(G)}{2} \log m
$$
式中 $\text{dim}(G)$ 为网络中自由独立参数的总数量。该公式极为直观：第一项为数据拟合优度，第二项为随样本量对数增长的模型复杂度惩罚项。

```julia
# 贝叶斯网络评分计算函数实现
function bayesian_score_component(M, α)
    p =  sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α,dims=2)))
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2)))
    return p
end

function bayesian_score(vars, G, D)
    n = length(vars)
    M = statistics(vars, G, D)
    α = prior(vars, G)
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n)
end
####################
```

---

## 5.2 有向图空间搜索算法 (Directed Graph Search)

包含 $n$ 个节点的可能 DAG 数量随着 $n$ 的增长呈现超指数爆炸（对于 $n=10$，可能的 DAG 数量超过 $4.2 \times 10^{18}$ 个！）。因此必须采用启发式搜索。

### 5.2.1 局部搜索算子与贪心爬山法
在有向图空间中，定义基础的局部邻域变换算子：
1. **加边（Add Edge）**：添加一条未出现的有向边 $X_i \to X_j$；
2. **删边（Delete Edge）**：删除一条已存在的有向边 $X_i \to X_j$；
3. **反转边（Reverse Edge）**：将已存在的 $X_i \to X_j$ 反转为 $X_j \to X_i$。

**核心约束**：执行任何变换后，所得图**必须严格保持无环性（Acyclic）**。贪心爬山算法在当前图的合法邻域中评估每个操作带来的评分增量，选择提升最大的操作前行，直到陷入局部极大值。

```julia
# 有向无环图局部邻域搜索实现
struct LocalDirectedGraphSearch
    G     # initial graph
    k_max # number of iterations
end

function rand_graph_neighbor(G)
    n = nv(G)
    i = rand(1:n)
    j = mod1(i + rand(2:n)-1, n)
    G′ = copy(G)
    has_edge(G, i, j) ? rem_edge!(G′, i, j) : add_edge!(G′, i, j)
    return G′
end

function fit(method::LocalDirectedGraphSearch, vars, D)
    G = method.G
    y = bayesian_score(vars, G, D)
    for k in 1:method.k_max
        G′ = rand_graph_neighbor(G)
        y′ = is_cyclic(G′) ? -Inf : bayesian_score(vars, G′, D)
        if y′ > y
            y, G = y′, G′
        end
    end
    return G
end
####################
```

### 5.2.2 K2 算法 (Cooper & Herskovits, 1992)
如果工程人员具备一定的领域先验，能够提供变量之间的一个**前向拓扑排序**（即保证若存在边 $X_i \to X_j$，则变量 $X_i$ 在排序中必定位于 $X_j$ 之前），那么结构搜索难度将大幅衰减！

**K2 算法原理**：
1. 固定节点顺序 $X_1, X_2, \dots, X_n$；
2. 每个节点 $X_i$ 只能从排在其前方的候选集 $\{X_1, \dots, X_{i-1}\}$ 中挑选父节点；
3. 对每个节点独立进行贪心前向选择：每次添加一个能使局部评分提升最大的候选父节点，直到父节点数量达到上限或评分不再增加。

```julia
# K2 结构学习算法实现
struct K2Search
    ordering::Vector{Int} # variable ordering
end

function fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y′ = bayesian_score(vars, G, D)
                    if y′ > y_best
                        y_best, j_best = y′, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else
                break
            end
        end
    end
    return G
end
####################
```

---

## 5.3 马尔可夫等价类 (Markov Equivalence Classes)

不同的 DAG 图拓扑可能编码完全相同的条件独立性断言集合。例如对于三个二值变量：
$$
X \to Y \to Z, \quad X \leftarrow Y \to Z, \quad X \leftarrow Y \leftarrow Z
$$
这三个网络在图拓扑上各不相同，但它们**均断言且仅断言 $(X \perp Z \mid Y)$**！任何基于观测统计数据的独立性测试都无法在它们之间进行区分。

### 5.3.1 I-等价性充分必要定理
若两个 DAG $G_1$ 与 $G_2$ 诱导完全相同的条件独立性假设集合，则称它们**马尔可夫等价（Markov Equivalent / I-Equivalent）**。

**判定定理（Verma & Pearl, 1990）**：两个 DAG 马尔可夫等价，当且仅当它们满足以下两个条件：
1. **拥有完全相同的骨架（Skeleton）**：即忽略边的方向后，无向连通图完全相同；
2. **拥有完全相同的非道德 v-结构（Immoral v-structures / Colliders）**：即所有形如 $X \to Y \leftarrow Z$ 且 $X$ 与 $Z$ 之间无直接边的倒置分叉结构完全相同。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_5_4.png" alt="道德与非道德 v-结构对比" style="max-width: 320px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 5.4：道德 v-结构（左，父节点间存在直接连边）与非道德 v-结构（右，父节点间无直接连边）。</p>
</div>

### 5.3.2 部分有向无环图 (PDAG)
一个马尔可夫等价类可以用一个**部分有向无环图（Partially Directed Acyclic Graph, PDAG / Essential Graph）**唯一表示：
- 若等价类中所有 DAG 在某条边上的方向均严格一致，则保留该有向边（表明存在确定性的因果碰撞导向）；
- 若等价类中存在不同方向的成员，则将其表示为**无向边**。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_5_5.png" alt="本质图与具体 DAG 成员" style="max-width: 680px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 5.5：一个包含 4 个变量的马尔可夫等价类本质图（左）及其代表的 4 个同构 DAG 成员与 1 个非法成员。</p>
</div>

---

## 5.4 贪心等价搜索 (Greedy Equivalence Search, GES)

直接在 DAG 空间中搜索存在明显的重复冗余，因为大量候选步只是在同一等价类的不同成员之间来回振荡。现代结构学习算法倾向于直接在**马尔可夫等价类空间（PDAG 空间）**中进行搜索。

最著名的两阶段算法是**贪心等价搜索（Greedy Equivalence Search, GES, Chickering, 2002）**：
1. **前向扩展阶段（Forward Phase）**：从全无向空图开始，系统评估向当前等价类中添加无向或有向边的评分增量，贪心执行提升最大的加边操作，直至评分不再增长；
2. **后向修剪阶段（Backward Phase）**：从前向阶段达到的局部极值出发，系统评估删除边的评分增量，贪心修剪冗余边，直至评分不再提升。

理论已严格证明：在样本量趋向无穷且数据生成分布忠实于某个 DAG 时，**GES 算法保证以概率 1 收敛至真实数据生成网络的马尔可夫等价类**！

---

## 5.5 本章小结 (Summary)

- **评分准则设计**：单纯最大化似然必然导致过拟合的全连接图；贝叶斯评分与 BIC 准则通过积分边际化内生施加了对自由参数规模的复杂度惩罚；
- **组合优化挑战**：DAG 搜索空间呈现超指数增长，是典型的 NP-hard 问题，局部邻域爬山算子与 K2 前向拓扑排序提供了实用的求解路径；
- **马尔可夫等价性本质**：仅凭静态观测数据无法区分骨架相同且非道德 v-结构相同的图，数据只能辨识到等价类本质图级别；
- **GES 算法的全局优越性**：通过在前向加边和后向删边两个阶段直接在等价类空间操作，具备渐近大样本下收敛至全局真实因果等价类的理论保证。

---

## 5.6 课后习题与官方详细解答 (Exercises & Solutions)

### 习题 5.1 (Exercise 5.1)
**题目**：对于一个包含 $m$ 个节点的完全无边图（Edgeless DAG），其局部搜索邻域中有多少个合法的相邻图结构？

**详细解答**：
局部搜索包含三种操作：加边、删边、反转边。
- 由于当前图没有任何边，删边与反转边的候选数为 0；
- 只能执行加边操作：在 $m$ 个节点之间可以添加任意一条有向边；
- 任意一对节点对 $(u, v)$（$u \neq v$）可以引入有向边 $u \to v$ 或 $v \to u$；
- 节点对组合数为 $\binom{m}{2} = \frac{m(m-1)}{2}$，每对有两种可能方向；
因此总共可以添加 $2 \times \frac{m(m-1)}{2} = \mathbf{m(m-1)}$ 条合法有向边，且单条边绝不可能构成环路。故合法邻居数为 $m(m-1)$。

---

### 习题 5.2 (Exercise 5.2)
**题目**：考虑包含 4 个节点的链式贝叶斯网络：$A \to B \to C \to D$。该图在局部搜索邻域中共有多少个合法的无环相邻网络？

**详细解答**：
4 节点完全图共有 $4 \times 3 = 12$ 条可能的有向边。当前图中已存在 3 条边：$A \to B$，$B \to C$，$C \to D$。
1. **删边操作**：可删除 3 条既有边中的任意一条，产生 3 个合法邻居；
2. **反转边操作**：
   - 反转 $A \to B$ 为 $B \to A$：不产生环路，合法；
   - 反转 $B \to C$ 为 $C \to B$：不产生环路，合法；
   - 反转 $C \to D$ 为 $D \to C$：不产生环路，合法；
   共产生 3 个合法邻居；
3. **加边操作**：剩余未出现的边共有 $12 - 3 = 9$ 条。
   - 添加前向跨级边：$A \to C$，$A \to D$，$B \to D$（3 条），均无环，合法；
   - 添加反向回边：$D \to C$（反向边），$C \to B$，$B \to A$ 这三条属于反转操作已讨论；
   - 检查反向长回边：$C \to A$（形成环路 $A \to B \to C \to A$，非法）、$D \to A$（形成环路，非法）、$D \to B$（形成环路，非法）；
   故合法的加边操作仅有 3 种；
综合汇总：$3 (\text{删边}) + 3 (\text{反转}) + 3 (\text{加边}) = \mathbf{9}$ 个合法相邻网络。

---

### 习题 5.3 (Exercise 5.3)
**题目**：设我们从某个初始网络 $G$ 启动局部爬山搜索。为了收敛到最优网络，最少需要执行多少次局部搜索迭代？

**详细解答**：
**最少需要 0 次迭代**。
若初始网络 $G$ 恰好已经是搜索空间中的局部最优解（即所有相邻邻域图的评分均严格低于 $G$），则算法在第 0 轮评估完所有邻居后便会立即终止并收敛。

---

### 习题 5.4 (Exercise 5.4)
**题目**：考虑由以下边构成的贝叶斯网络：$A \to C \leftarrow B$，$C \to D$。画出表示其马尔可夫等价类的部分有向无环图（PDAG），并求该等价类中总共包含多少个具体的有向无环图？

**详细解答**：
1. **寻找非道德 v-结构**：
   - 节点 $C$ 拥有父节点 $A$ 与 $B$（$A \to C \leftarrow B$），且 $A$ 与 $B$ 之间没有连边。这是一个非道德 v-结构！根据等价类判定定理，该倒置分叉的方向必须严格固定；
2. **检查剩余边 $C \to D$**：
   - 若将 $C \to D$ 反转为 $D \to C$：节点 $C$ 将拥有三个父节点 $A, B, D$。此时在父节点对 $(A, D)$ 和 $(B, D)$ 之间没有连边，这会凭空引入两个全新的非道德 v-结构（$A \to C \leftarrow D$ 和 $B \to C \leftarrow D$），从而破坏了原有的条件独立性集合！
   - 因此，边 $C \to D$ 的方向同样是唯一确定的，不能反转；
3. **结论**：该马尔可夫等价类中的所有边方向均为强制固定的，其 PDAG 即为其自身，该等价类中**仅包含 1 个**具体的 DAG。

---

### 习题 5.5 (Exercise 5.5)
**题目**：给出一个包含 4 个节点的半有向图示例，证明其不能对应任何非空的合法马尔可夫等价类。

**详细解答**：
考虑 4 个节点排成环状无向图：$A - B - C - D - A$（无弦四边形环）。
如果试图为该无向四边形的所有边指定有向方向以构成 DAG：
- 如果分配方向使其不产生环路，必然在某个拐角处引入至少一个未加保护的非道德 v-结构（例如 $A \to B \leftarrow C$）；
- 然而，一旦引入该 v-结构，根据等价类的定义，该 v-结构的有向边必须在 PDAG 中显式标记为有向箭头，这与初始设定的全无向边假设产生不可调和的逻辑矛盾！
因此，长度大于等于 4 的无弦环状半有向图不能表示任何合法的马尔可夫等价类。
