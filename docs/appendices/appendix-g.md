# 附录 G：Julia 语言速查 (The Julia Language)

本书所有算法均采用 **Julia 语言** 编写。Julia 巧妙化解了科学计算中著名的“两语言问题（Two-Language Problem）”——既具备 Python、MATLAB 等高级动态语言如数学公式般的优雅表达力，又具备 C、Fortran 等底层编译语言的纯正原生机器码执行性能。

---

## G.1 类型系统与类型树 (Type Hierarchy)

Julia 拥有完备的参数化多态抽象类型系统。所有具体类型均隶属于某种抽象父类型：

<div style="text-align: center; margin: 20px 0;">
  <img src="/figures/fig_G_1.png" alt="Julia Float64 类型层次树" style="max-width: 300px; display: inline-block; border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);" />
  <p style="color: #666; font-size: 0.9em; margin-top: 6px;">图 G.1：Julia 中标准浮点数 `Float64` 溯源至 `AbstractFloat`、`Real`、`Number` 直至 `Any` 的抽象类型继承树。</p>
</div>

---

## G.2 核心特性与多重分派 (Multiple Dispatch)

1. **多重分派（Multiple Dispatch）**：函数的执行分支不仅取决于调用主体，而是由**参数签名列表中所有实参的具体类型组合在运行时动态精确匹配**，天然契合数学上多变量运算的扩展性需求；
2. **向量化广播（Broadcasting）**：通过简洁的句点语法 `f.(x)` 或 `x .+ y`，自动将任意标量函数高效向量化应用于任意高维数组，并由底层 LLVM 自动完成 SIMD 指令循环融合；
3. **原生无开销抽象**：不可变结构体（`struct`）在内存中采用紧凑连续布局，完全消除了指针间接寻址开销。
