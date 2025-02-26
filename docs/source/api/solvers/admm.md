# Introduction

## Dual problem

Consider the constrained convex optimization problem:

$$
\begin{aligned}
    &\text{minimize} \quad f(x) \\
    &\text{subject to} \quad Ax = b
\end{aligned}
$$

- Lagrangian: $L(x, y) = f(x) + y^T (Ax - b)$
- Dual function: $g(y) = \inf_x L(x, y)$
- Dual problem: $\text{maximize} \quad g(y)$
- Recover $x^\star = \arg\min_x L(x, y^\star)$

## Dual ascent

To solve the dual problem, we used the Gradient Descent method:

$$
y_{k+1} = y_k + \alpha_k \nabla g(y_k)
$$

It is easy to see that:

$$
\nabla g(y_k) = Ax^\ast - b \\
x^\ast = \arg\min_x L(x, y_k)
$$

Therefore, we have update rules to solve the problem:

$$
\begin{aligned}
    x_{k+1} &:= \arg\min_x L(x, y_k) \\
    y_{k+1} &:= y_k + \alpha_k (Ax_{k+1} - b)
\end{aligned}
$$

We update until $Ax_{k+1} - b \rightarrow 0$.

## Method of Multipliers

Powell introduced **augmented Lagrangian**, with hyperparameter $\rho > 0$:

$$
L_{\rho}(x, y) = f(x) + y^T (Ax - b) + (\rho / 2) \|Ax - b\|_2^2
$$

With the new **augmented Lagrangian**, we have new update rules:

$$
\begin{aligned}
    x_{k+1} &:= \arg\min_x L_{\rho}(x, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} - b)
\end{aligned}
$$

We update until $Ax_{k+1} - b \rightarrow 0$.

## Alternating Direction Method of Multipliers (ADMM)

ADMM problem's form:

$$
\begin{aligned}
    &\text{minimize} \quad f(x) + g(z) \\
    &\text{subject to} \quad Ax + Bz = c
\end{aligned}
$$

The **augmented Lagrangian**, with $\rho > 0$:

$$
L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz - c) + (\rho / 2) \|Ax + Bz - c\|_2^2
$$

Instead of solving $x, z$ jointly, we solve them separately (Gauss-Seidel method). Therefore, update rules are:

$$
\begin{aligned}
    x_{k+1} &:= \arg\min_x L_{\rho}(x, z_k, y_k)\\
    z_{k+1} &:= \arg\min_z L_{\rho}(x_{k+1}, z, y_k) \\
    y_{k+1} &:= y_k + \rho (Ax_{k+1} + Bz_{k+1} - c)
\end{aligned}
$$

We update until $Ax_k + Bz_k - c \to 0$.

# Implicit Deep Learning

Instead of using recursive formulas as in traditional neural networks, we consider implicit prediction rules:

$$
\begin{aligned}
    x &= \phi(Ax + Bu) \quad \text{[equilibrium equation]} \\
    \hat{y}(u) &= Cx + Du \quad \text{[prediction equation]}
\end{aligned}
$$

## State-driven Implicit Modeling

### Training

To achieve better compression with minimal performance loss, we solve:

$$
\begin{aligned}
    & \min_{M} \quad f(M)\\
    & \text{s.t.} \quad
    Z = AX + BU, \\
    & \quad \quad \hat{Y} = CX + DU, \\
    & \quad \quad \|A\|_\infty \leq \kappa.
\end{aligned}
$$

where $M = \begin{pmatrix} A & B \\ C & D \end{pmatrix}$.

### Inference

After training, given $U$, solve $X = \sigma(AX + BU)$, then compute $\hat{Y} = CX + DU$.

## SIM Training Setup

We minimize the max function using the $l_\infty$ norm:

$$
\min_{a,b} \|a\|_1 + \|b\|_1 + \lambda_1 \|z - X^T a - U^T b\|_2^2
$$

where each row is solved independently.

# ADMM in SIM

Using the ADMM method, we solve:

$$
\min_{a,b} \frac{1}{2} \|X^T a + U^T b - z\|_2^2 + \lambda \|a, b\|_1
$$

using consensus form and iterative updates.

# References

- Powell, "A method for nonlinear constraints in minimization problems," 1969.
- Tsai et al., "State-driven Implicit Modeling for Sparsity and Robustness in Neural Networks," 2022.
- Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers," 2011.


# Bibtex file

```texttext
@misc{tsai2022statedrivenimplicitmodelingsparsity,
  title = {State-driven Implicit Modeling for Sparsity and Robustness in Neural Networks},
  author = {Alicia Y. Tsai and Juliette Decugis and Laurent El Ghaoui and Alper Atamtürk},
  year = {2022},
  eprint = {2209.09389},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2209.09389}
}


@book{10.5555/2834535,
author = {Hastie, Trevor and Tibshirani, Robert and Wainwright, Martin},
title = {Statistical Learning with Sparsity: The Lasso and Generalizations},
year = {2015},
isbn = {1498712169},
publisher = {Chapman \& Hall/CRC},
abstract = {Discover New Methods for Dealing with High-Dimensional Data A sparse statistical model has only a small number of nonzero parameters or weights; therefore, it is much easier to estimate and interpret than a dense model. Statistical Learning with Sparsity: The Lasso and Generalizations presents methods that exploit sparsity to help recover the underlying signal in a set of data. Top experts in this rapidly evolving field, the authors describe the lasso for linear regression and a simple coordinate descent algorithm for its computation. They discuss the application of 1 penalties to generalized linear models and support vector machines, cover generalized penalties such as the elastic net and group lasso, and review numerical methods for optimization. They also present statistical inference methods for fitted (lasso) models, including the bootstrap, Bayesian methods, and recently developed approaches. In addition, the book examines matrix decomposition, sparse multivariate analysis, graphical models, and compressed sensing. It concludes with a survey of theoretical results for the lasso. In this age of big data, the number of features measured on a person or object can be large and might be larger than the number of observations. This book shows how the sparsity assumption allows us to tackle these problems and extract useful and reproducible patterns from big datasets. Data analysts, computer scientists, and theorists will appreciate this thorough and up-to-date treatment of sparse statistical modeling.}
}



@misc{ghaoui2020implicitdeeplearning,
      title={Implicit Deep Learning}, 
      author={Laurent El Ghaoui and Fangda Gu and Bertrand Travacca and Armin Askari and Alicia Y. Tsai},
      year={2020},
      eprint={1908.06315},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1908.06315}, 
}

@article{boyd2011distributed,
  title={Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers},
  author={Boyd, Stephen and Parikh, Neal and Chu, Eric and Peleato, Borja and Eckstein, Jonathan},
  journal={Foundations and Trends® in Machine Learning},
  volume={3},
  number={1},
  pages={1--122},
  year={2011},
  publisher={Now Publishers, Inc.}
}

@inproceedings{Powell1969AMF,
  title={A method for nonlinear constraints in minimization problems},
  author={M. J. D. Powell},
  year={1969},
  url={https://api.semanticscholar.org/CorpusID:115810962}
}
```