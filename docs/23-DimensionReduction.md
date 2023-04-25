# (PART) Dimension Reduction Methods {-}

# Matrix Decompositions {-}
  
Matrix decomposition, also known as matrix factorization, is a process of breaking down a matrix into simpler components that can be used to simplify calculations, solve systems of equations, and gain insight into the underlying structure of the matrix. 
  
Matrix decomposition plays an important role in machine learning, particularly in the areas of dimensionality reduction, data compression, and feature extraction. For example, Principal Component Analysis (PCA) is a popular method for dimensionality reduction, which involves decomposing a high-dimensional data matrix into a lower-dimensional representation while preserving the most important information. PCA achieves this by finding the eigenvectors and eigenvalues of the covariance matrix of the data and then selecting the top eigenvectors as the new basis for the data.

Singular Value Decomposition (SVD) is also commonly used in recommender systems to find latent features in user-item interaction data. SVD decomposes the user-item interaction matrix into three matrices: a left singular matrix, a diagonal matrix of singular values, and a right singular matrix. The left and right singular matrices represent user and item features, respectively, while the singular values represent the importance of those features.

Rank optimization is another method that finds a low-rank approximation of a matrix that best fits a set of observed data. In other words, it involves finding a lower-rank approximation of a given matrix that captures the most important features of the original matrix.  For example, SVD decomposes a matrix into a product of low-rank matrices, while PCA finds the principal components of a data matrix, which can be used to create a lower-dimensional representation of the data.   In machine learning, rank optimization is often used in applications such as collaborative filtering, image processing, and data compression. By finding a low-rank approximation of a matrix, it is possible to reduce the amount of memory needed to store the matrix and improve the efficiency of algorithms that work with the matrix.

We start with the eigenvalue decomposition (EVD), which is the foundation to many matrix decomposition methods

# Eigenvectors and eigenvalues  

Eigenvalues and eigenvectors have many important applications in linear algebra and beyond. For example, in machine learning, principal component analysis (PCA) involves computing the eigenvectors and eigenvalues of the covariance matrix of a data set, which can be used to reduce the dimensionality of the data while preserving its important features. 

Almost all vectors change direction, when they are multiplied by a matrix, $\mathbf{A}$, except for certain vectors ($\mathbf{v}$) that are in the same direction as $\mathbf{A} \mathbf{v}.$ Those vectors are called "eigenvectors". 

We can see how we obtain the eigenvalues and eigenvectors of a matrix $\mathbf{A}$. If

$$
\mathbf{A} \mathbf{v}=\lambda \mathbf{v}
$$
 
Then,
   
$$
\begin{aligned}
&\mathbf{A} \mathbf{v}-\lambda \mathbf{I} \mathbf{v}=0 \\
&(\mathbf{A}-\lambda \mathbf{I}) \mathbf{v}=0,
\end{aligned}
$$
where $\mathbf{I}$ is the identity matrix. It turns out that this equation is equivalent to:

$$
\operatorname{det}(\mathbf{A}-\lambda \mathbf{I})=0,
$$

because $\operatorname{det}(\mathbf{A}-\lambda \mathbf{I}) \equiv(\mathbf{A}-\lambda \mathbf{I}) \mathbf{v}=0$.  The reason is that we want a non-trivial solution to $(\mathbf{A}-\lambda \mathbf{I}) \mathbf{v}=0$.  Therefore, $(\mathbf{A}-\lambda \mathbf{I})$ should be non-invertible. Otherwise, if it is invertible, we get $\mathbf{v}=(\mathbf{A}-\lambda \mathbf{I})^{-1} \cdot 0=0$, which is a trivial solution. Since a matrix is non-invertible if its determinant is 0 . Thus, $\operatorname{det}(\mathbf{A}-\lambda \mathbf{I})=0$ for non-trivial solutions.

We start with a square matrix, $\mathbf{A}$, like

$$
A =\left[\begin{array}{cc}
1 & 2 \\
3 & -4
\end{array}\right]
$$
$$
\begin{aligned}
\det (\mathbf{A}-\lambda \mathbf{I})=
& \left|\begin{array}{cc}
1-\lambda & 2 \\
3 & -4-\lambda
\end{array}\right|=(1-\lambda)(-4-\lambda)-2 \cdot 3 \\
& =-4-\lambda+4 \lambda+\lambda^2-6 \\
& =\lambda^2+3 \lambda-10 \\
& =(\lambda-2)(\lambda+5)=0 \\
& \therefore \lambda_1=2, ~ \lambda_2=-5 \\
&
\end{aligned}
$$

We have two eigenvalues.  We now need to consider each eigenvalue indivudally

$$
\begin{gathered}
\lambda_1=2 \\
(A 1-\lambda I) \mathbf{v}=0 \\
{\left[\begin{array}{cc}
1-\lambda_1 & 2 \\
3 & -4-\lambda_1
\end{array}\right]\left[\begin{array}{l}
v_1 \\
v_2
\end{array}\right]=\left[\begin{array}{l}
0 \\
0
\end{array}\right]} \\

{\left[\begin{array}{cc}
-1 & 2 \\
3 & -6
\end{array}\right]\left[\begin{array}{l}
v_1 \\
v_2
\end{array}\right]=\left[\begin{array}{l}
0 \\
0
\end{array}\right]}
\end{gathered}
$$
Hence, 

$$
\begin{aligned}
-v_1+2 v_2=0 \\
3 v_1-6 v_2=0\\
v_1=2, ~ v_2=1
\end{aligned}
$$
And,

$$
\begin{aligned}
&  \lambda_2=-5 \\
& {\left[\begin{array}{cc}
1-\lambda_2 & 2 \\
3 & -4-\lambda_2
\end{array}\right]\left[\begin{array}{l}
v_1 \\
v_2
\end{array}\right]=\left[\begin{array}{l}
0 \\
0
\end{array}\right]} \\
& {\left[\begin{array}{cc}
6 & 2 \\
3 & 1
\end{array}\right]\left[\begin{array}{l}
v_1 \\
v_2
\end{array}\right]=\left[\begin{array}{l}
0 \\
0
\end{array}\right]} 

\end{aligned}
$$
Hence, 

$$
\begin{gathered}
6 v_1+2 v_2=0 \\
3 v_1+v_2=0 \\

v_1=-1,~ v_2=3
\end{gathered}
$$
We have two eigenvalues

$$
\begin{aligned}
& \lambda_1=2 \\
& \lambda_2=-5
\end{aligned}
$$

And two corresponding eigenvectors

$$
\left[\begin{array}{l}
2 \\
1
\end{array}\right],\left[\begin{array}{c}
-1 \\
3
\end{array}\right]
$$
for $\lambda_1=2$

$$
\left[\begin{array}{cc}
1 & 2 \\
3 & -4
\end{array}\right]\left[\begin{array}{l}
2 \\
1
\end{array}\right]=\left[\begin{array}{l}
2+2 \\
6-4
\end{array}\right]=\left[\begin{array}{l}
4 \\
2
\end{array}\right]=2\left[\begin{array}{l}
2 \\
1
\end{array}\right]
$$
Let's see the solution in R


```r
A <- matrix(c(1, 3, 2, -4), 2, 2)
eigen(A)
```

```
## eigen() decomposition
## $values
## [1] -5  2
## 
## $vectors
##            [,1]      [,2]
## [1,] -0.3162278 0.8944272
## [2,]  0.9486833 0.4472136
```

The eigenvectors are typically normalized by dividing by its length $\sqrt{v^{\prime} v}$, which is 5 in our case for $\lambda_1=2$.


```r
# For the ev (2, 1), for lambda
c(2, 1) / sqrt(5)
```

```
## [1] 0.8944272 0.4472136
```

There some nice properties that we can observe in this application.


```r
# Sum of eigenvalues = sum of diagonal terms of A (Trace of A)
ev <- eigen(A)$values
sum(ev) == sum(diag(A))
```

```
## [1] TRUE
```

```r
# Product of eigenvalues = determinant of A
round(prod(ev), 4) == round(det(A), 4)
```

```
## [1] TRUE
```

```r
# Diagonal matrix D has eigenvalues = diagonal elements
D <- matrix(c(2, 0, 0, 5), 2, 2)
eigen(D)$values == sort(diag(D), decreasing = TRUE)
```

```
## [1] TRUE TRUE
```

We can see that, if one of the eigenvalues is zero for a matrix, the determinant of the matrix will be zero.  We willl return to this issue in Singluar Value Decomposition.

Let's finish this chapter with Diagonalization and Eigendecomposition.

Suppose we have $m$ linearly independent eigenvectors ($\mathbf{v_i}$ is eigenvector $i$ in a column vector in $\mathbf{V}$) of $\mathbf{A}$.

$$
\mathbf{AV}=\mathbf{A}\left[\mathbf{v_1} \mathbf{v_2} \cdots \mathbf{v_m}\right]=\left[\mathbf{A} \mathbf{v_1} \mathbf{A} \mathbf{v_2} \ldots \mathbf{A} \mathbf{v_m}\right]=\left[\begin{array}{llll}
\lambda_1 \mathbf{v_1} & \lambda_2\mathbf{v_2}  & \ldots & \lambda_m \mathbf{v_m}
\end{array}\right]
$$

because 

$$
\mathbf{A} \mathbf{v}=\lambda \mathbf{v}
$$

$$
\mathbf{AV}=\left[\mathbf{v_1} \mathbf{v_2} \cdots \mathbf{v_m}\right]\left[\begin{array}{cccc}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\

0 & 0 & \cdots & \lambda_m
\end{array}\right]=\mathbf{V}\Lambda
$$
So that

$$
\mathbf{A V=V \Lambda}
$$
Hence,   

$$
\mathbf{A}=\mathbf{V} \Lambda \mathbf{V}^{-1}
$$

Eigendecomposition (a.k.a. spectral decomposition) decomposes a matrix $\mathbf{A}$ into a multiplication of a matrix of eigenvectors $\mathbf{V}$ and a diagonal matrix of eigenvalues $\mathbf{\Lambda}$.
  
**This can only be done if a matrix is diagonalizable**. In fact, the definition of a diagonalizable matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is that it can be eigendecomposed into $n$ eigenvectors, so that $\mathbf{V}^{-1} \mathbf{A} \mathbf{V}=\Lambda$.

$$
\begin{align}
\mathbf{A}^2&=(\mathbf{V} \Lambda \mathbf{V}^{-1})(\mathbf{V} \Lambda \mathbf{V}^{-1})\\
&=\mathbf{V} \Lambda \text{I} \Lambda \mathbf{V}^{-1}\\
&=\mathbf{V} \Lambda^2 \mathbf{V}^{-1}\\
\end{align}
$$
in general

$$
\mathbf{A}^k=\mathbf{V} \Lambda^k \mathbf{V}^{-1}
$$

Example:
  

```r
A = matrix(sample(1:100, 9), 3, 3)
A
```

```
##      [,1] [,2] [,3]
## [1,]   19    6    4
## [2,]   54   45   79
## [3,]   33   59   17
```

```r
eigen(A)
```

```
## eigen() decomposition
## $values
## [1] 105.88436 -38.78498  13.90062
## 
## $vectors
##            [,1]        [,2]       [,3]
## [1,] 0.08268915  0.02213879  0.8139177
## [2,] 0.81588495 -0.69337367 -0.4350121
## [3,] 0.57227113  0.72023804 -0.3851006
```

```r
V = eigen(A)$vectors
Lam = diag(eigen(A)$values)
# Prove that AV = VLam
round(A %*% V, 4) == round(V %*% Lam, 4)
```

```
##      [,1] [,2] [,3]
## [1,] TRUE TRUE TRUE
## [2,] TRUE TRUE TRUE
## [3,] TRUE TRUE TRUE
```

```r
# And decomposition
A == round(V %*% Lam %*% solve(V), 4)
```

```
##      [,1] [,2] [,3]
## [1,] TRUE TRUE TRUE
## [2,] TRUE TRUE TRUE
## [3,] TRUE TRUE TRUE
```

And, matrix inverse with eigendecomposition:

$$
\mathbf{A}^{-1}=\mathbf{V} \Lambda^{-1} \mathbf{V}^{-1}
$$

Example:
  

```r
A = matrix(sample(1:100, 9), 3, 3)
A
```

```
##      [,1] [,2] [,3]
## [1,]   56   30   13
## [2,]   90   32   81
## [3,]   20   51   61
```

```r
V = eigen(A)$vectors
Lam = diag(eigen(A)$values)

# Inverse of A
solve(A)
```

```
##             [,1]         [,2]        [,3]
## [1,]  0.01166651  0.006248193 -0.01078309
## [2,]  0.02072023 -0.016897427  0.01802178
## [3,] -0.02114855  0.012078769  0.00486149
```

```r
# And
V %*% solve(Lam) %*% solve(V)
```

```
##             [,1]         [,2]        [,3]
## [1,]  0.01166651  0.006248193 -0.01078309
## [2,]  0.02072023 -0.016897427  0.01802178
## [3,] -0.02114855  0.012078769  0.00486149
```

The inverse of $\mathbf{\Lambda}$ is just the inverse of each diagonal element (the eigenvalues).  But, this can only be done if a matrix is diagonalizable.  So if $\mathbf{A}$ is not $n \times n$, then we can use $\mathbf{A'A}$ or $\mathbf{AA'}$, both symmetric now.

Example:
$$
\mathbf{A}=\left(\begin{array}{ll}
1 & 2 \\
2 & 4
\end{array}\right)
$$

As $\det(\mathbf{A})=0,$ $\mathbf{A}$ is singular and its inverse is undefined.  In other words, since $\det(\mathbf{A})$ equals the product of the eigenvalues $\lambda_j$ of $\mathrm{A}$, the matrix $\mathbf{A}$ has an eigenvalue which is zero.

To see this, consider the spectral (eigen) decomposition of $A$ :
$$
\mathbf{A}=\sum_{j=1}^{p} \theta_{j} \mathbf{v}_{j} \mathbf{v}_{j}^{\top}
$$
where $\mathbf{v}_{\mathrm{j}}$ is the eigenvector belonging to $\theta_{\mathrm{j}}$

The inverse of $\mathbf{A}$ is then:
  
$$
\mathbf{A}^{-1}=\sum_{j=1}^{p} \theta_{j}^{-1} \mathbf{v}_{j} \mathbf{v}_{j}^{\top}
$$

A has eigenvalues 5 and 0. The inverse of $A$ via the spectral decomposition is then undefined:
  
$$
\mathbf{A}^{-1}=\frac{1}{5} \mathbf{v}_{1} \mathbf{v}_{1}^{\top}+ \frac{1}{0} \mathbf{v}_{1} \mathbf{v}_{1}^{\top}
$$
