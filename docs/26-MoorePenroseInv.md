# Moore-Penrose inverse 

Another example is solving OLS with SVD:
  
$$
\mathbf{y = X \beta}\\
\mathbf{y = U \Sigma V' \beta}\\
\mathbf{U'y = U'U \Sigma V' \beta}\\
\mathbf{U'y = \Sigma V' \beta}\\
\mathbf{\Sigma^{-1}}\mathbf{U'y =  V' \beta}\\
\mathbf{V\Sigma^{-1}}\mathbf{U'y =  \beta}\\
$$
  
And 

$$
\mathbf{V\Sigma^{-1}U' = M^+}
$$
is called **"generalized inverse" or The Moore-Penrose Pseudoinverse**.  
  
Here are some application of SVD and Pseudoinverse.  


```r
library(MASS)

##Simple SVD and generalized inverse
a <- matrix(c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1), 9, 4)

a.svd <- svd(a)
ds <- diag(1/a.svd$d[1:3])
u <- a.svd$u
v <- a.svd$v
us <- as.matrix(u[, 1:3])
vs <- as.matrix(v[, 1:3])
(a.ginv <- vs %*% ds %*% t(us))
```

```
##             [,1]        [,2]        [,3]        [,4]        [,5]        [,6]
## [1,]  0.08333333  0.08333333  0.08333333  0.08333333  0.08333333  0.08333333
## [2,]  0.25000000  0.25000000  0.25000000 -0.08333333 -0.08333333 -0.08333333
## [3,] -0.08333333 -0.08333333 -0.08333333  0.25000000  0.25000000  0.25000000
## [4,] -0.08333333 -0.08333333 -0.08333333 -0.08333333 -0.08333333 -0.08333333
##             [,7]        [,8]        [,9]
## [1,]  0.08333333  0.08333333  0.08333333
## [2,] -0.08333333 -0.08333333 -0.08333333
## [3,] -0.08333333 -0.08333333 -0.08333333
## [4,]  0.25000000  0.25000000  0.25000000
```

```r
ginv(a)
```

```
##             [,1]        [,2]        [,3]        [,4]        [,5]        [,6]
## [1,]  0.08333333  0.08333333  0.08333333  0.08333333  0.08333333  0.08333333
## [2,]  0.25000000  0.25000000  0.25000000 -0.08333333 -0.08333333 -0.08333333
## [3,] -0.08333333 -0.08333333 -0.08333333  0.25000000  0.25000000  0.25000000
## [4,] -0.08333333 -0.08333333 -0.08333333 -0.08333333 -0.08333333 -0.08333333
##             [,7]        [,8]        [,9]
## [1,]  0.08333333  0.08333333  0.08333333
## [2,] -0.08333333 -0.08333333 -0.08333333
## [3,] -0.08333333 -0.08333333 -0.08333333
## [4,]  0.25000000  0.25000000  0.25000000
```

```r
##Simulated DGP
x1 <- rep(1, 20)
x2 <- rnorm(20)
x3 <- rnorm(20)
u <- matrix(rnorm(20, mean=0, sd=1), nrow=20, ncol=1)
X <- cbind(x1, x2, x3)
beta <- matrix(c(0.5, 1.5, 2), nrow=3, ncol=1)
Y <- X%*%beta + u

##OLS
betahat_OLS <- solve(t(X)%*%X)%*%t(X)%*%Y
betahat_OLS
```

```
##         [,1]
## x1 0.6970372
## x2 1.3931879
## x3 1.7876766
```

```r
##SVD
X.svd <- svd(X)
ds <- diag(1/X.svd$d)
u <- X.svd$u
v <- X.svd$v
us <- as.matrix(u)
vs <- as.matrix(v)
X.ginv_mine <- vs %*% ds %*% t(us)

# Compare
X.ginv <- ginv(X)
round((X.ginv_mine - X.ginv),4)
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12] [,13] [,14]
## [1,]    0    0    0    0    0    0    0    0    0     0     0     0     0     0
## [2,]    0    0    0    0    0    0    0    0    0     0     0     0     0     0
## [3,]    0    0    0    0    0    0    0    0    0     0     0     0     0     0
##      [,15] [,16] [,17] [,18] [,19] [,20]
## [1,]     0     0     0     0     0     0
## [2,]     0     0     0     0     0     0
## [3,]     0     0     0     0     0     0
```

```r
# Now OLS
betahat_ginv <- X.ginv %*%Y
betahat_ginv
```

```
##           [,1]
## [1,] 0.6970372
## [2,] 1.3931879
## [3,] 1.7876766
```

```r
betahat_OLS
```

```
##         [,1]
## x1 0.6970372
## x2 1.3931879
## x3 1.7876766
```
  
Now the question where and when we can use `ginv`? With a high-dimensional $\mathbf{X}$, where $p > n$, the vector $\beta$ cannot uniquely be determined from the system of equations.the solution to the normal equation is 

$$
\hat{\boldsymbol{\beta}}=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{+} \mathbf{X}^{\top} \mathbf{Y}+\mathbf{v} \quad \text { for all } \mathbf{v} \in \mathcal{V}
$$

where $\mathbf{A}^{+}$denotes the Moore-Penrose inverse of the matrix $\mathbf{A}$. Therefore, there is no unique estimator of the regression parameter (See Page 7 for proof in [Lecture notes on ridge regression](https://arxiv.org/pdf/1509.09169.pdf)) [@Wieringen_2021].  To arrive at a unique regression estimator for studies with rank deficient design matrices, the minimum least squares estimator may be employed.

The minimum least squares estimator of regression parameter minimizes the sum-of-squares criterion and is of minimum length. Formally, $\hat{\boldsymbol{\beta}}_{\mathrm{MLS}}=\arg \min _{\boldsymbol{\beta} \in \mathbb{R}^{p}}\|\mathbf{Y}-\mathbf{X} \boldsymbol{\beta}\|_{2}^{2}$ such that $\left\|\hat{\boldsymbol{\beta}}_{\mathrm{MLS}}\right\|_{2}^{2}<\|\boldsymbol{\beta}\|_{2}^{2}$ for all $\boldsymbol{\beta}$ that minimize
$\|\mathbf{Y}-\mathbf{X} \boldsymbol{\beta}\|_{2}^{2}$.

So $\hat{\boldsymbol{\beta}}_{\mathrm{MLS}}=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{+} \mathbf{X}^{\top} \mathbf{Y}$ is the minimum least squares estimator of regression parameter minimizes the sum-of-squares criterion.  
  
As we talked before in Chapter 17, an alternative (and related) estimator of the regression parameter $\beta$ that avoids the use of the Moore-Penrose inverse and is able to deal with (super)-collinearity among the columns of the design matrix is the ridge regression estimator proposed by [Hoerl and Kennard (1970)](https://www.math.arizona.edu/~hzhang/math574m/Read/RidgeRegressionBiasedEstimationForNonorthogonalProblems.pdf). They propose to simply replace $\mathbf{X}^{\top} \mathbf{X}$ by $\mathbf{X}^{\top} \mathbf{X}+\lambda \mathbf{I}_{p p}$ with $\lambda \in[0, \infty)$.  The ad-hoc fix solves the singularity as it adds a positive matrix, $\lambda \mathbf{I}_{p p}$, to a positive semi-definite one, $\mathbf{X}^{\top} \mathbf{X}$, making the total a positive definite matrix, which is invertible.

Hence, the ad-hoc fix of the ridge regression estimator resolves the non-evaluation of the estimator in the face of super-collinearity but yields a 'ridge fit' that is not optimal in explaining the observation. Mathematically, this is due to the fact that the fit $\widehat{Y}(\lambda)$ corresponding to the ridge regression estimator is not a projection of $Y$ onto the covariate space.
