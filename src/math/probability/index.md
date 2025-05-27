# 概率

研究客观世界中的不确定现象（随机现象）。

如何研究？

对于某个**不确定现象**，我们可以通过建模选定一个**样本空间** \\(\Omega\\)，并在其上构造一个 \\(\sigma\\)-代数 \\(\mathcal{F}\\)，从而形成可定义**事件**的结构。进一步引入一个**概率测度** \\(\mathbb{P}\\)，使得三元组\\((\Omega, \mathcal{F}, \mathbb{P})\\) 成为一个**概率空间**。
在此基础上，我们可以定义从 \\(\Omega\\) 到其他测度空间（例如实数集 \\(\mathbb{R}\\) 及其 Borel \\(\sigma\\)-代数 \\(\mathcal{B}(\mathbb{R})\\)）的可测函数，称为**随机变量**（同一个概率空间可以定义多个不一样的随机变量）。即
\\[
X : \Omega \to \mathcal{B}
\\]
满足对任意 Borel 集合 \\(B \in \mathcal{B}(\mathbb{R})\\)，有
\\[
X^{-1}(B) = \{\omega \in \Omega : X(\omega) \in B\} \in \mathcal{F}.
\\]

每个随机变量在概率测度 \\(\mathbb{P}\\) 下诱导出一个**概率分布函数**：
\\[
F_X(x) = \mathbb{P}(X \leq x), \quad x \in \mathbb{R}.
\\]

该分布函数由随机变量 \\(X\\) 和概率测度 \\(\mathbb{P}\\) 联合**唯一确定**。

有了随机变量的概率分布函数后，我们可以计算其各种统计特性，以及一些收敛性质，例如：

- **期望**（数学期望）：
\\[
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \, dF_X(x)
\\]

若 \\(X\\) 有概率密度函数 \\(f_X(x)\\)，则写成
\\[
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x f_X(x) \, dx
\\]

- **方差**：
\\[
\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \int_{-\infty}^{+\infty} (x - \mathbb{E}[X])^2 \, dF_X(x)
\\]
