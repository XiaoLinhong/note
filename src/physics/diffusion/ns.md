
描述流体运动最完整、最准确的方法当属偏微分方程（PDE）

流体动力方程组包含质量守恒方程、动量守恒方程和能量守恒方程

通过雷诺输运定理推导三个方程的保守形式。

雷诺输运定理的基本形式如下：

\\[
\begin{equation}
\frac{d}{dt} \int_{V(t)} \psi \, dV = 
\int_{V(t)} \frac{\partial \psi}{\partial t} dV + 
\int_{A(t)} \mathbf{v} \psi \cdot d\mathbf{A}
\end{equation}
\\]

其中：
  - \( \psi \) 是任意的流体强度变量（可以是密度、动量密度、能量密度等），\( \psi = \rho B\)
  - \( V(t) \) 是随流体运动的控制体（即拉格朗日控制体）
  - \( A(t) \) 是该控制体的边界面，\( d\mathbf{A} \) 是指向外的微分面积矢量。
  - \( \mathbf{v} \cdot d\mathbf{A} \) 表示流体通过控制体表面的通量（流体穿过控制体的速率）。

根据高斯散度定理
\\[
  \int_A \mathbf{F} \cdot d\mathbf{A} = \int_V \nabla \cdot \mathbf{F} \, dV
\\]

其中
  - \( A \) 是封闭曲面，\( d\mathbf{A} \) 是指向外的单位法向面积元素
  - \( V \) 是被 \( A \) 包围的体积
  - \( \mathbf{F} \) 是连续可微的向量场
  - \( \nabla \cdot \mathbf{F} \) 是 \( \mathbf{F} \) 的散度。

因此雷诺输运定理可以改写为完全的体积分形式：

\\[
\frac{d}{dt} \int_{V(t)} \psi \, dV = 
\int_{V(t)} \left( \frac{\partial \psi}{\partial t} + 
\nabla \cdot (\psi \mathbf{v}) \right) dV.
\\]


与物质导数不一样，雷诺输运定理的左边的系统体积，是一个面积可以变化的流体微元。
注意在使用雷诺输运定理，数值计算时，一般使用的是等式的右边（欧拉法），
等式的左边为全导数，为拉格朗日法，
便于我们应用物理定律（受力分析时，当作质点，能量分析时，当作孤立的系统）。

连续方程， 是比质量守恒更强的约束，质量在空间中处处守恒。

\\[
    \frac{\partial \rho}{ \partial t} + \nabla \cdot (\mathbf{v} \rho) = 0
\\]

动量定律，可以表述为动量变化量等于它在这个过程中所受力的冲量。
流体微元受到的作用力包括气压梯度力、黏性力、重力和旋转参考系中存在科氏力


\\[
    \frac{\partial \rho \mathbf{v}}{\partial t} + \nabla \cdot \mathbf{v} \rho\mathbf{v} = -
     \nabla p + \nabla \cdot \boldsymbol{\tau} + \rho \mathbf{g}
\\]
其中 \(\boldsymbol{\tau}\) 为切应力压强（粘性力压强），为二阶张量


流体在运动中，需要保持能量守恒。
流体微元包含3种能量，包括内能（注意内能的温度表示，用的是定容比热！一种静态的表示方式），
动能和重力势能，不考虑重力势能，
因为重力只能导致机械能的转换，也就是说只在重力的作用下

\\[
      e = C_v T + \frac{1}{2} \mathbf{v^2}
\\]

那么能量守恒公式可以写作

\\[
       \frac{\partial \rho e}{\partial t} +
    \nabla \cdot (\mathbf{v} \rho e) = 
    K \nabla^2 T + Q + 
    \nabla \cdot (-p \mathbf{v} + \boldsymbol{\tau} \cdot \mathbf{v})
\\]

为了使得方程闭合，引入状态方程
\\[
    p = \rho R T
\\]

以上四个方程，未知数为$(\mathbf{v}, \rho, p, T)$，因此，方程组式封闭的。
