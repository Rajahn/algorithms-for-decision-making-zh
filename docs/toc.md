# 《决策算法》全书总目录

> 原著：Mykel J. Kochenderfer, Tim A. Wheeler, Kyle H. Wray  
> 出版：MIT Press (2022) · [官方主页](https://algorithmsbook.com/) · [GitHub 代码库](https://github.com/algorithmsbooks/decisionmaking-code)

---

## 导言与第一部分：概率推理 (Probabilistic Reasoning) · 全卷上线

- **第 1 章：绪论 (Introduction)** [[已上线]](/part1/01-introduction)
  - 1.1 决策制定与智能体-环境交互回路
  - 1.2 六大工程应用（飞机防撞、自动驾驶、医疗筛查、金融消费投资、野火巡检、火星漫游车）
  - 1.3 核心方法体系（显式编程、监督学习、优化、规划、强化学习）
  - 1.4 跨学科发展简史（经济学、心理学、神经科学、计算机科学、工程学、数学、运筹学）
  - 1.5 社会影响与算法伦理
  - 1.6 全书架构概览与技术脉络导图
- **第 2 章：表示 (Representation)** [[已上线]](/part1/02-representation)
  - 2.1 信念程度与概率公理（柯尔莫哥洛夫公理、考克斯定理、荷兰赌论证）
  - 2.2 概率分布（PMF、PDF、CDF、分位数函数、截断高斯分布、多模态混合模型）
  - 2.3 联合概率分布（离散因式表、Assignment、维度灾难、多元高斯协方差矩阵）
  - 2.4 条件概率分布（贝叶斯法则、连续线性高斯模型、Logit / Softmax 跃迁曲面）
  - 2.5 贝叶斯网络（DAG 拓扑、局部条件因式分解定理、卫星遥测故障监控案例）
  - 2.6 条件独立性与 d-分离（因果链、共因分叉、倒置分叉碰撞节点、合理解释效应、d-分离判定）
  - 2.7 本章小结与 10 道课后习题全解
- **第 3 章：推断 (Inference)** [[已上线]](/part1/03-inference)
  - 3.1 贝叶斯网络中的推断任务（查询变量、证据变量、隐藏变量全概率展开）
  - 3.2 朴素贝叶斯模型（星形拓扑与板块表示、类条件独立性推断）
  - 3.3 和-积变量消除法（因子乘积、因子求和边际化、条件化切片、消除顺序与诱导树宽）
  - 3.4 信度传播（树形消息传递算法、Loopy BP 与 Bethe 自由能近似）
  - 3.5 计算复杂度（3-SAT 多项式规约与 NP-hard 证明）
  - 3.6 直接采样法与拒绝采样（前向拓扑采样及其在罕见事件下的失效缺陷）
  - 3.7 似然加权采样（固定证据与似然累乘赋权、无浪费蒙特卡洛积分）
  - 3.8 吉布斯采样（MCMC 马尔可夫链、马尔可夫毯局部全条件更新、收敛性对比）
  - 3.9 多元高斯模型中的精确推断（块矩阵条件分解、舒尔补闭式解）
  - 3.10 本章小结与 7 道课后习题全解
- **第 4 章：参数学习 (Parameter Learning)** [[已上线]](/part1/04-parameter-learning)
  - 4.1 极大似然参数学习（对数似然函数、充分统计量、离散经验频数统计、连续高斯参数估计）
  - 4.2 贝叶斯参数学习（共轭先验、Beta-Binomial 模型、Dirichlet-Multinomial 伪计数、拉普拉斯平滑、MAP vs 期望）
  - 4.3 非参数学习（核密度估计 KDE、高斯核函数、平滑带宽选择影响）
  - 4.4 缺失数据下的学习（MCAR、MAR、MNAR 三大缺失机制、插补技术对比、期望最大化 EM 算法）
  - 4.5 本章小结与 7 道课后习题全解
- **第 5 章：结构学习 (Structure Learning)** [[已上线]](/part1/05-structure-learning)
  - 5.1 贝叶斯网络评分准则（似然过拟合困境、贝叶斯边际似然评分、BIC/MDL 渐近准则、奥卡姆剃刀惩罚）
  - 5.2 有向图搜索算法（加边/删边/反转局部爬山算子、K2 拓扑排序启发式算法）
  - 5.3 马尔可夫等价类（I-等价性判定定理：相同骨架 + 相同非道德 v-结构、本质图 PDAG / CPDAG 表示）
  - 5.4 贪心等价搜索（GES 算法：前向加边阶段与后向删边阶段）
  - 5.5 本章小结与 5 道课后习题全解
- **第 6 章：简单决策 (Simple Decisions)** [[已上线]](/part1/06-simple-decisions)
  - 6.1 理性偏好的约束公理（彩票模型、冯·诺依曼-摩根斯坦六大公理）
  - 6.2 效用函数（正仿射变换不变性、风险态度：中性/厌恶/喜好、确定性等价值、风险溢价、幂效用函数）
  - 6.3 效用诱导（标准博弈法 Standard Gamble）
  - 6.4 最大期望效用原理（MEU 准则）
  - 6.5 决策网络 / 影响图（偶然节点、决策节点、效用节点、变量消除求解算法）
  - 6.6 信息价值（VOI / VPI 完全信息价值数学定义、非负性定理）
  - 6.7 人类非理性与行为经济学（阿莱悖论、埃尔斯伯格悖论、前景理论与损失厌恶）
  - 6.8 本章小结与 7 道课后习题全解

---

## 第二部分：序贯问题 (Sequential Problems) `[即将上线]`

- **第 7 章：精确解法 (Exact Solution Methods)**
  - 马尔可夫决策过程 (MDP) 形式化定义（$S, A, T, R, \gamma$）
  - 贝尔曼期望方程与贝尔曼最优性方程
  - 策略评估、策略迭代（Policy Iteration）、价值迭代（Value Iteration）
- **第 8 章：近似价值函数 (Approximate Value Functions)**
- **第 9 章：在线规划 (Online Planning)**
- **第 10 章：策略搜索 (Policy Search)**
- **第 11 章：策略梯度估计 (Policy Gradient Estimation)**
- **第 12 章：策略梯度优化 (Policy Gradient Optimization)**
- **第 13 章：演员-评论员方法 (Actor-Critic Methods)**
- **第 14 章：策略验证 (Policy Validation)**

---

## 第三部分：模型不确定性 (Model Uncertainty) `[待发布]`

- 第 15 章：探索与利用 (Exploration and Exploitation)
- 第 16 章：基于模型的方法 (Model-Based Methods)
- 第 17 章：无模型方法 (Model-Free Methods)
- 第 18 章：模仿学习 (Imitation Learning)

---

## 第四部分：状态不确定性 (State Uncertainty) `[待发布]`

- 第 19 章：置信状态 (Beliefs)
- 第 20 章：精确置信状态规划 (Exact Belief State Planning)
- 第 21 章：离线置信状态规划 (Offline Belief State Planning)
- 第 22 章：在线置信状态规划 (Online Belief State Planning)
- 第 23 章：控制器抽象 (Controller Abstractions)

---

## 第五部分：多智能体系统 (Multiagent Systems) `[待发布]`

- 第 24 章：多智能体推理 (Multiagent Reasoning)
- 第 25 章：序贯博弈 (Sequential Problems)
- 第 26 章：状态不确定性 (State Uncertainty)
- 第 27 章：协同智能体 (Collaborative Agents)

---

## 附录 (Appendices) `[待发布]`

- 附录 A：数学概念
- 附录 B：概率分布速查
- 附录 C：计算复杂度
- 附录 D：神经网络表示
- 附录 E：搜索算法
- 附录 F：基准问题库
- 附录 G：Julia 语言速查
