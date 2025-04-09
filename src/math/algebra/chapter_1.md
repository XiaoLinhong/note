# 线性代数

## **线性映射**  
在两个向量空间之间的映射 \\( T: V \to W \\) 被称为**线性映射**（Linear Map）或**线性变换**（Linear Transformation），如果它满足以下两个条件，对任意 \\( \mathbf{u}, \mathbf{v} \in V \\) 和标量 \\( a \in \mathbb{F} \\)：

1. **加法保持性（Additivity）**：
   \\[
   T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})
   \\]

2. **数乘保持性（Homogeneity）**：
   \\[
   T(a \mathbf{u}) = a T(\mathbf{u})
   \\]

即，线性映射保持向量加法和数乘运算。  

## **等价定义**
线性映射 \\( T: V \to W \\) 可以等价地表示为：
\\[
T(a \mathbf{u} + b \mathbf{v}) = a T(\mathbf{u}) + b T(\mathbf{v}), \quad \forall \mathbf{u}, \mathbf{v} \in V, \forall a, b \in \mathbb{F}
\\]
即，线性映射保持**线性组合**的运算。

## **特殊情况**
- 如果 \\( V = W \\)，则 \\( T \\) 是从自身到自身的**线性变换**（线性算子，有时候一个意思，有时候有所区分）。
- 如果 \\( T \\) 映射到数域 \\( \mathbb{F} \\)（即 \\( W = \mathbb{F} \\)），则 \\( T \\) 是一个**线性函数（比如 多元一次函数）**。

## **矩阵表示**
如果 \\( V \\) 和 \\( W \\) 都是有限维向量空间，则线性映射可以用**矩阵** \\( A \\) 表示（不能有高次，最后就写到了多元一次函数方程组）：
\\[
T(\mathbf{x}) = A \mathbf{x}
\\]
其中，\\( A \\) 是一个 \\( m \times n \\) 矩阵，\\( \mathbf{x} \\) 是 \\( n \\) 维列向量，\\( T(\mathbf{x}) \\) 是 \\( m \\) 维列向量。

## **例子**
1. **简单的数乘变换**：
   \\( T: \mathbb{R}^2 \to \mathbb{R}^2 \\)，定义为 \\( T(x, y) = (2x, 3y) \\)，满足线性性，因此是线性映射。

2. **矩阵变换**：
   设 \\( A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \\)，则映射 \\( T(\mathbf{x}) = A \mathbf{x} \\) 是线性映射。

3. **积分算子**：
   设 \\( V \\) 是所有连续函数的空间，定义 \\( T(f) = \int_0^x f(t) \, dt \\)，可以验证 \\( T \\) 也是线性映射。
