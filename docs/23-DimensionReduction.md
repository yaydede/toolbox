# (PART) Dimension Reduction Methods {-}

# Matrix Decompositions {-}
  
Any matrix decomposition (and related topics) requires a solid understanding of eigenvalues and singular value decomposition (SVD).  

# Eigenvectors and eigenvalues  

To explain eigenvalues, we first explain eigenvectors. Almost all vectors change direction, when they are multiplied by $\mathbf{A}$. Certain exceptional vectors $x$ are in the same direction as $\mathbf{A} x .$ Those are the "eigenvectors". Multiply an eigenvector by $\mathbf{A}$, and the vector $\mathbf{A}x$ is a number $\lambda$ times the original $x$.
  
The basic equation is $\mathbf{A} x=\lambda x$. The number $\lambda$ is an eigenvalue of $\mathbf{A}$. The eigenvalue $\lambda$ tells whether the special vector $x$ is stretched or shrunk or reversed or left unchanged-when it is multiplied by $\mathbf{A}$. We may find $\lambda=2$ or $\frac{1}{2}$ or $-1$ or 1. The eigenvalue $\lambda$ could be zero! Then $\mathbf{A} x=0 x$ means that this eigenvector $x$ is in the nullspace.

A good example comes from the powers $\mathbf{A, A^{2}, A^{3}}, \ldots$ of a matrix. Suppose you need the hundredth power $\mathbf{A}^{100}$. The starting matrix $\mathbf{A}$ becomes unrecognizable after a few steps, and $\mathbf{A}^{100}$ is very close to $\left[\begin{array}{llll}.6 & .6 ; & .4 & .4\end{array}\right]:$

$$
\begin{aligned}
&{\left[\begin{array}{cc}
.8 & .3 \\
.2 & .7
\end{array}\right] \quad\left[\begin{array}{cc}
.70 & .45 \\
.30 & .55
\end{array}\right] \quad\left[\begin{array}{cc}
.650 & .525 \\
.350 & .475
\end{array}\right] \ldots} & {\left[\begin{array}{ll}
.6000 & .6000 \\
.4000 & .4000
\end{array}\right]} \\
&~~~~\mathbf{A} ~~~~~~~~~~~~~~~~~~~~~~~~~\mathbf{A}^{2}~~~~~~~~~~~~~~~~~~~~\mathbf{A}^{3}& \mathbf{A}^{100}
\end{aligned}
$$
  
$\mathbf{A}^{100}$ was found by using the eigenvalues of $\mathbf{A}$, not by multiplying 100 matrices.
   
Those eigenvalues (here they are 1 and $1 / 2$ ) are a new way to see into the heart of a matrix. See <http://math.mit.edu/~gs/linearalgebra/ila0601.pdf> [@Strang_2016] for more details.  Most $2 \times 2$ matrices have two eigenvector directions and two eigenvalues and it can be shown that $\operatorname{det}(\mathbf{A}-\lambda I)=0$.


```r
#*****Eigenvalues and vectors*******
#AX = lambdaX
#(A − lambdaI)X = 0

A <- matrix(c(2,1,8,5), 2, 2)
ev <- eigen(A)$values

# Sum of eigenvalues = sum of diagonal terms of A (called the trace of A)
sum(ev)
```

```
## [1] 7
```

```r
sum(diag(A))
```

```
## [1] 7
```

```r
# Product of eigenvalues = determinant of A
prod(ev)
```

```
## [1] 2
```

```r
det(A)
```

```
## [1] 2
```

```r
# Diagonal matrix D has eigenvalues = diagonal elements
D <- matrix(c(2,0,0,5), 2, 2)
eigen(D)
```

```
## eigen() decomposition
## $values
## [1] 5 2
## 
## $vectors
##      [,1] [,2]
## [1,]    0   -1
## [2,]    1    0
```

$\text{Rank}(\mathbf{A})$ is number of nonzero singular values of $\mathbf{A}$. Singular values are eigenvalues of $\mathbf{X'X}$ which is an $n \times n$ matrix and $\mathbf{X}$ is $m \times n$ matrix.  SVD starts with eigenvalue decomposition.  See <http://www.onmyphd.com/?p=eigen.decomposition>.  

Eigendecomposition is the method to decompose a square matrix into its eigenvalues and eigenvectors. For a matrix $\mathbf{A}$, if

$$
\mathbf{A} \mathbf{v}=\lambda \mathbf{v}
$$
  
then $\mathbf{v}$ is an eigenvector of matrix $\mathbf{A}$ and $\lambda$ is the corresponding eigenvalue. That is, if matrix $\mathbf{A}$ is multiplied by a vector and the result is a scaled version of the same vector, then it is an eigenvector of $\mathbf{A}$ and the scaling factor is its eigenvalue.
  
So how do we find the eigenvectors of a matrix? 

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

because $\operatorname{det}(\mathbf{A}-\lambda \mathbf{I}) \equiv(\mathbf{A}-\lambda \mathbf{I}) \mathbf{v}=0$.
  
Why? Since you want non-trivial solutions to $(\mathbf{A}-\lambda \mathbf{I}) \mathbf{v}=0$, you want $(\mathbf{A}-\lambda \mathbf{I})$ to be non-invertible. Otherwise, its invertible and you get $\mathbf{v}=(\mathbf{A}-\lambda \mathbf{I})^{-1} \cdot 0=0$ which is a trivial solution. But a linear transformation or a matrix is non-invertible if and only if its determinant is 0 . So $\operatorname{det}(\mathbf{A}-\lambda \mathbf{I})=0$ for non-trivial solutions.
  
It's hard to understand the intuition or why eigenvectors and values are important. Here is the excerpt from [How to intuitively understand eigenvalue and eigenvector](https://math.stackexchange.com/q/243553) [@Use_eigen]:
  
<style type="text/css">
blockquote {
    padding: 10px 20px;
    margin: 0 0 20px;
    font-size: 14px;
    border-left: 5px solid #eee;
}
</style>

> First let us think what a square matrix does to a vector. Consider a matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$. Let us see what the matrix $\mathbf{A}$ acting on a vector $x$ does to this vector. By action, we mean multiplication i.e. we get a new vector $y=\mathbf{A} x$.
>
> The matrix acting on a vector $x$ does two things to the vector $x$. (1) It scales the vector; (2) It rotates the vector. It is important to understand what the matrix $\mathbf{A}$ in a set of equations $\mathbf{A x}=\mathbf{b}$ does. Matrix $\mathbf{A}$ simply "transforms" a vector $\mathbf{x}$ into another vector $\mathbf{b}$ by applying linear combination. The transformation is done within the same space or subspace. Sometimes we only want to know what would be the vector $\mathbf{b}$ if linear combination is applied, that is when we execute the equation $\mathbf{A x}=\mathbf{b}$. Other times we are interested in a reverse problem and we want to solve the equation $\mathbf{x}=\mathbf{A}^{-1} \mathbf{b}$.
>
> However, for any matrix $\mathbf{A}$, there are some favored vectors/directions. When the matrix acts on these favored vectors, the action essentially results in just scaling the vector. There is no rotation. These favored vectors are precisely the eigenvectors and the amount by which each of these favored vectors stretches or compresses is the eigenvalue.
>

So why are these eigenvectors and eigenvalues important? Consider the eigenvector corresponding to the maximum (absolute) eigenvalue. If we take a vector along this eigenvector, then the action of the matrix is maximum. No other vector when acted by this matrix will get stretched as much as this eigenvector.

Hence, if a vector were to lie "close" to this eigen direction, then the "effect" of action by this matrix will be "large" i.e. the action by this matrix results in "large" response for this vector. The effect of the action by this matrix is high for large (absolute) eigenvalues and less for small (absolute) eigenvalues. Hence, the directions/vectors along which this action is high are called the principal directions or principal eigenvectors. The corresponding eigenvalues are called the principal values.

Here are some examples:

$$
\mathbf{\Lambda}=\left[\begin{array}{cc}
\lambda_{1} & 0 \\
0 & \lambda_{2}
\end{array}\right] \\
$$
$$
\mathbf{V}=\left[\mathbf{v}_1 \mathbf{v}_2\right]
$$
So that

$$
\mathbf{A V=V \Lambda}
$$
Hence,   
$$
\mathbf{A=V \Lambda V^{-1}}
$$

Eigendecomposition (a.k.a. spectral decomposition) decomposes a matrix $\mathbf{A}$ into a multiplication of a matrix of eigenvectors $\mathbf{V}$ and a diagonal matrix of eigenvalues $\mathbf{\Lambda}$.
  
**This can only be done if a matrix is diagonalizable**. In fact, the definition of a diagonalizable matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is that it can be eigendecomposed into $n$ eigenvectors, so that $\mathbf{V^{-1} A V=\Lambda}$.

Example:
  

```r
A = matrix(sample(1:100, 9), 3, 3)
A
```

```
##      [,1] [,2] [,3]
## [1,]   33   23   65
## [2,]   34   28   67
## [3,]   83   70   47
```

```r
eigen(A)
```

```
## eigen() decomposition
## $values
## [1] 153.418840 -47.681644   2.262803
## 
## $vectors
##            [,1]       [,2]        [,3]
## [1,] -0.4818493 -0.4779009 -0.65821115
## [2,] -0.5109215 -0.4520648  0.75146612
## [3,] -0.7118852  0.7531588  0.04535146
```

```r
V = eigen(A)$vectors
Lam = diag(eigen(A)$values)
# Prove that AV = V lambda and 
A%*%V
```

```
##            [,1]      [,2]       [,3]
## [1,]  -73.92476  22.78710 -1.4894024
## [2,]  -78.38498  21.55519  1.7004200
## [3,] -109.21660 -35.91185  0.1026214
```

```r
V%*%Lam
```

```
##            [,1]      [,2]       [,3]
## [1,]  -73.92476  22.78710 -1.4894024
## [2,]  -78.38498  21.55519  1.7004200
## [3,] -109.21660 -35.91185  0.1026214
```

```r
# And decomposition
V%*%Lam%*%solve(V)
```

```
##      [,1] [,2] [,3]
## [1,]   33   23   65
## [2,]   34   28   67
## [3,]   83   70   47
```

And, matrix inverse with eigendecomposition

$$
\mathbf{A^{-1}=V \Lambda^{-1} V^{-1}}
$$

Example:
  

```r
A = matrix(sample(1:100, 9), 3, 3)
A
```

```
##      [,1] [,2] [,3]
## [1,]   11   12   81
## [2,]   64   94   68
## [3,]   39   69   30
```

```r
V = eigen(A)$vectors
Lam = diag(eigen(A)$values)

# Inverse of A
solve(A)
```

```
##             [,1]         [,2]         [,3]
## [1,] -0.03824936  0.106840750 -0.138899105
## [2,]  0.01495648 -0.057803114  0.090637898
## [3,]  0.01532426 -0.005945813  0.005435005
```

```r
# And
V%*%solve(Lam)%*%solve(V)
```

```
##                [,1]            [,2]            [,3]
## [1,] -0.03824936+0i  0.106840750+0i -0.138899105+0i
## [2,]  0.01495648+0i -0.057803114-0i  0.090637898+0i
## [3,]  0.01532426+0i -0.005945813-0i  0.005435005-0i
```

The inverse of $\mathbf{\Lambda}$ is just the inverse of each diagonal element (the eigenvalues).  But, **this can only be done if a matrix is diagonalizable**.  So if $\mathbf{A}$ is not $n \times n$, then we can use $\mathbf{A'A}$ or $\mathbf{AA'}$, both symmetric now.

Example:
$$
\mathbf{A}=\left(\begin{array}{ll}
1 & 2 \\
2 & 4
\end{array}\right)
$$

As $\operatorname{det}(\mathbf{A})=0, \mathbf{A}$ is singular and its inverse is undefined. $\operatorname{Det}(\mathbf{A})$ equals the product of the eigenvalues $\theta_{\mathrm{j}}$ of $\mathrm{A}$: the matrix $\mathbf{A}$ is singular if any eigenvalue of $\mathbf{A}$ is zero.

To see this, consider the spectral decomposition of $A$ :
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
