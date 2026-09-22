# 附录 D：神经网络表示 (Neural Representations)

人工神经网络（Artificial Neural Networks）是现代深度强化学习、连续价值函数逼近与端到端控制的核心非线性表示工具。

---

## D.1 全连接层与前向传播 (Fully Connected Layers)

一个全连接前馈神经网络层（Dense Layer）通过仿射线性变换叠加逐元素非线性激活函数构造：
$$
\mathbf{x}' = \sigma(\mathbf{W} \mathbf{x} + \mathbf{b})
$$

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_D_1.png" alt="全连接层神经元连线" style="max-width: 380px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 D.1：包含 3 个输入与 2 个神经元的全连接层微观连线拓扑。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_D_4.png" alt="双层网络分类分界面" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 D.2：两层全连接神经网络在二维输入平面上拟合出的非线性决策分界面。</p>
  </div>
</div>

---

## D.2 常用非线性激活函数族 (Activation Functions)

若没有非线性激活函数，任意深度的多层网络在代数上仅仅等价于单个单层线性矩阵乘法。

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_D_5.png" alt="常见非线性激活函数曲线对比" style="max-width: 480px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 D.3：常用激活函数曲线对比：Sigmoid、双曲正切 Tanh、ReLU、Leaky ReLU 与平滑 Softplus 函数。</p>
</div>

1. **Sigmoid**：$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$，常用于概率输出；
2. **Tanh**：$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$，零均值中心化；
3. **ReLU**：$\max(0, x)$，有效解决深度反向传播中的梯度消失（Vanishing Gradient）；
4. **Softplus**：$\log(1 + e^x)$，ReLU 的光滑可微解析平滑版。

---

## D.3 循环网络与生成对抗架构 (RNN & GAN)

对于部分可观测 POMDP 或时间序列决策，智能体需要具有时序记忆功能。

<div style="display: flex; justify-content: center; gap: 24px; margin: 20px 0; flex-wrap: wrap;">
  <div style="text-align: center;">
    <img src="/figures/fig_D_10.png" alt="RNN 时序展开图" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 D.4：循环神经网络（RNN）沿时间轴向前无损展开的时序数据流图。</p>
  </div>
  <div style="text-align: center;">
    <img src="/figures/fig_D_13.png" alt="生成对抗网络架构" style="max-width: 320px; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
    <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 D.5：生成对抗网络（GAN）中生成器与判别器双向对抗博弈架构。</p>
  </div>
</div>

循环神经网络通过隐状态 $h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$ 维持历史记忆；而生成对抗网络（GAN）则为第 18 章的 GAIL 模仿学习奠定了判别器对抗对偶机制。
