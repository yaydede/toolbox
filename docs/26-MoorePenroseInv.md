# Moore-Penrose inverse 

The Singular Value Decomposition (SVD) can be used for solving Ordinary Least Squares (OLS) problems. In particular, the SVD of the design matrix $\mathbf{X}$ can be used to compute the coefficients of the linear regression model.  Here are the steps:
  
$$
\mathbf{y = X \beta}\\
\mathbf{y = U \Sigma V' \beta}\\
\mathbf{U'y = U'U \Sigma V' \beta}\\
\mathbf{U'y = \Sigma V' \beta}\\
\mathbf{\Sigma^{-1}}\mathbf{U'y =  V' \beta}\\
\mathbf{V\Sigma^{-1}}\mathbf{U'y =  \beta}\\
$$

This formula for beta is computationally efficient and numerically stable, even for ill-conditioned or singular $\mathbf{X}$ matrices. Moreover, it allows us to compute the solution to the OLS problem without explicitly computing the inverse of $\mathbf{X}^T \mathbf{X}$. 

Menawhile, the term

$$
\mathbf{V\Sigma^{-1}U' = M^+}
$$

is called **"generalized inverse" or The Moore-Penrose Pseudoinverse**.  

If $\mathbf{X}$ has full column rank, then the pseudoinverse is also the unique solution to the OLS problem. However, if $\mathbf{X}$ does not have full column rank, then its pseudoinverse may not exist or may not be unique. In this case, the OLS estimator obtained using the pseudoinverse will be a "best linear unbiased estimator" (BLUE), but it will not be the unique solution to the OLS problem.

To be more specific, the OLS estimator obtained using the pseudoinverse will minimize the sum of squared residuals subject to the constraint that the coefficients are unbiased, i.e., they have zero expected value. However, there may be other linear unbiased estimators that achieve the same minimum sum of squared residuals. These alternative estimators will differ from the OLS estimator obtained using the pseudoinverse in the values they assign to the coefficients.

In practice, the use of the pseudoinverse to estimate the OLS coefficients when $\mathbf{X}$ does not have full column rank can lead to numerical instability, especially if the singular values of $\mathbf{X}$ are very small. In such cases, it may be more appropriate to use regularization techniques such as ridge or Lasso regression to obtain stable and interpretable estimates. These methods penalize the size of the coefficients and can be used to obtain sparse or "shrunken" estimates, which can be particularly useful in high-dimensional settings where there are more predictors than observations.


Here are some application of SVD and Pseudoinverse.  


```r
library(MASS)

##Simple SVD and generalized inverse
A <- matrix(c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1), 9, 4)

a.svd <- svd(A)
ds <- diag(1 / a.svd$d[1:3])
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
ginv(A)
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

We can use SVD for solving a regular OLS on simulated data:  


```r
#Simulated DGP
x1 <- rep(1, 20)
x2 <- rnorm(20)
x3 <- rnorm(20)
u <- matrix(rnorm(20, mean = 0, sd = 1), nrow = 20, ncol = 1)
X <- cbind(x1, x2, x3)
beta <- matrix(c(0.5, 1.5, 2), nrow = 3, ncol = 1)
Y <- X %*% beta + u

#OLS
betahat_OLS <- solve(t(X) %*% X) %*% t(X) %*% Y
betahat_OLS
```

```
##         [,1]
## x1 0.4187644
## x2 1.6604130
## x3 1.3810833
```

```r
#SVD
X.svd <- svd(X)
ds <- diag(1 / X.svd$d)
u <- X.svd$u
v <- X.svd$v
us <- as.matrix(u)
vs <- as.matrix(v)
X.ginv_mine <- vs %*% ds %*% t(us)

# Compare
X.ginv <- ginv(X)
round((X.ginv_mine - X.ginv), 4)
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
betahat_ginv <- X.ginv %*% Y
betahat_ginv
```

```
##           [,1]
## [1,] 0.4187644
## [2,] 1.6604130
## [3,] 1.3810833
```

```r
betahat_OLS
```

```
##         [,1]
## x1 0.4187644
## x2 1.6604130
## x3 1.3810833
```
  
