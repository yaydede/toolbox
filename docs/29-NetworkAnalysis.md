# (PART) Network Analysis {-}

# Graphical Network Analysis {-}
  
A network represents a structure of relationships between objects. Graphical modeling presents a network structure in a graph, which consists of nodes and edges, by expressing conditional (in)dependence between the nodes. If we think of these nodes (objects) as variables and their relationship with each other as edges, a graphical model represents the probabilistic relationships among a set of variables. For example, the absence of edges (partial correlations) corresponds to conditional independence. Graphical models are becoming more popular in statistics because it helps us understand a very complex structure of relationships in networks, such as the dynamic structure of biological systems or social events.  

The central idea is that, since any pair of nodes may be joined by an edge, a missing edge represents some form of independency between the pair of variables. The complexity in network analysis comes from the fact that the independency may be either marginal or conditional on some or all of the other variables.  Therefore, defining a graphical model requires identification of a type of graph needed for each particular case.

In general, a graphical model could be designed with directed and undirected edges. In a directed graph, an arrow indicates the direction of dependency between nodes. In undirected graphs, however, the edges do not have directions.  The field of graphical modeling is vast, hence it is beyond the scope of this book. 

Yet, we will look at the precision matrix, which has been shown that its regularization captures the network connections.  Hence, the central theme of this section is the estimation of sparse standardized precision matrices, whose results can be illustrated by undirected graphs.

# Fundementals

In this chapter, we will cover several concepts related to statistical (in)dependence measured by correlations.

## Covariance

We start with a data matrix, which refers to the array of numbers:  

$$
\mathbf{X}=\left(\begin{array}{cccc}
x_{11} & x_{12} & \cdots & x_{1 p} \\
x_{21} & x_{22} & \cdots & x_{2 p} \\
x_{31} & x_{32} & \cdots & x_{3 p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n 1} & x_{n 2} & \cdots & x_{n p}
\end{array}\right)
$$

An example would be


```r
set.seed(5)
x <- rnorm(30, sd=runif(30, 2, 50))
X <- matrix(x, 10)
X
```

```
##              [,1]       [,2]       [,3]
##  [1,]   -1.613670  -4.436764  42.563842
##  [2,]  -20.840548  36.237338 -36.942481
##  [3,] -100.484392  25.903897 -24.294407
##  [4,]    3.769073 -18.950442 -22.616651
##  [5,]   -1.821506 -12.454626  -1.243431
##  [6,]   32.103933   3.693050  38.807102
##  [7,]   25.752668  22.861071 -18.452338
##  [8,]   59.864792  98.848864  -3.607105
##  [9,]   33.862342  34.853324  16.704375
## [10,]    5.980194  62.755408 -21.841795
```

We start with defining the covariance matrix

$$
\mathbf{S}=\left(\begin{array}{ccccc}
s_{1}^{2} & s_{12} & s_{13} & \cdots & s_{1 p} \\
s_{21} & s_{2}^{2} & s_{23} & \cdots & s_{2 p} \\
s_{31} & s_{32} & s_{3}^{2} & \cdots & s_{3 p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
s_{p 1} & s_{p 2} & s_{p 3} & \cdots & s_{p}^{2}
\end{array}\right)
$$
$$
s_{j}^{2}=(1 / n) \sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)^{2} 
$$
is the variance of the $j$-th variable,

$$
\begin{aligned}
&s_{j k}=(1 / n) \sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)\left(x_{i k}-\bar{x}_{k}\right) 
\end{aligned}
$$
is the covariance between the $j$-th and $k$-th variables; and, 

$$
\bar{x}_{j}=(1 / n) \sum_{i=1}^{n} x_{j i} 
$$
is the mean of the $j$-th variable.

We can calculate the covariance matrix such as

$$
\mathbf{S}=\frac{1}{n} \mathbf{X}_{c}^{\prime} \mathbf{X}_{c},
$$

where $\mathbf{X}_{c}$ is the centered matrix: 

$$
\mathbf{X}_{c}=\left(\begin{array}{cccc}
x_{11}-\bar{x}_{1} & x_{12}-\bar{x}_{2} & \cdots & x_{1 p}-\bar{x}_{p} \\
x_{21}-\bar{x}_{1} & x_{22}-\bar{x}_{2} & \cdots & x_{2 p}-\bar{x}_{p} \\
x_{31}-\bar{x}_{1} & x_{32}-\bar{x}_{2} & \cdots & x_{3 p}-\bar{x}_{p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n 1}-\bar{x}_{1} & x_{n 2}-\bar{x}_{2} & \cdots & x_{n p}-\bar{x}_{p}
\end{array}\right)
$$

How?


```r
# More direct
n <- nrow(X)
m <- matrix(1, n, 1)%*%colMeans(X)
Xc <- X-m
Xc
```

```
##               [,1]        [,2]        [,3]
##  [1,]   -5.2709585 -29.3678760  45.6561309
##  [2,]  -24.4978367  11.3062262 -33.8501919
##  [3,] -104.1416804   0.9727849 -21.2021184
##  [4,]    0.1117842 -43.8815539 -19.5243622
##  [5,]   -5.4787951 -37.3857380   1.8488577
##  [6,]   28.4466449 -21.2380620  41.8993911
##  [7,]   22.0953790  -2.0700407 -15.3600493
##  [8,]   56.2075038  73.9177518  -0.5148158
##  [9,]   30.2050530   9.9222117  19.7966643
## [10,]    2.3229057  37.8242961 -18.7495065
```

```r
# Or
C <- diag(n) - matrix(1/n, n, n)
XC <- C %*% X
Xc
```

```
##               [,1]        [,2]        [,3]
##  [1,]   -5.2709585 -29.3678760  45.6561309
##  [2,]  -24.4978367  11.3062262 -33.8501919
##  [3,] -104.1416804   0.9727849 -21.2021184
##  [4,]    0.1117842 -43.8815539 -19.5243622
##  [5,]   -5.4787951 -37.3857380   1.8488577
##  [6,]   28.4466449 -21.2380620  41.8993911
##  [7,]   22.0953790  -2.0700407 -15.3600493
##  [8,]   56.2075038  73.9177518  -0.5148158
##  [9,]   30.2050530   9.9222117  19.7966643
## [10,]    2.3229057  37.8242961 -18.7495065
```

```r
# We can also use `scale` 
Xc <- scale(X, center=TRUE, scale=FALSE)
```

And, the covariance matrix


```r
# Covariance Matrix
S <- t(Xc) %*% Xc / (n-1)
S
```

```
##           [,1]      [,2]      [,3]
## [1,] 1875.3209  429.8712  462.4775
## [2,]  429.8712 1306.9817 -262.8231
## [3,]  462.4775 -262.8231  755.5193
```

```r
# Check it
cov(X)
```

```
##           [,1]      [,2]      [,3]
## [1,] 1875.3209  429.8712  462.4775
## [2,]  429.8712 1306.9817 -262.8231
## [3,]  462.4775 -262.8231  755.5193
```

## Correlation

While covariance is a necessary step, we can capture the size and the direction of relationships between the variables:

$$
\mathbf{R}=\left(\begin{array}{ccccc}
1 & r_{12} & r_{13} & \cdots & r_{1 p} \\
r_{21} & 1 & r_{23} & \cdots & r_{2 p} \\
r_{31} & r_{32} & 1 & \cdots & r_{3 p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
r_{p 1} & r_{p 2} & r_{p 3} & \cdots & 1
\end{array}\right)
$$

where

$$
r_{j k}=\frac{s_{j k}}{s_{j} s_{k}}=\frac{\sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)\left(x_{i k}-\bar{x}_{k}\right)}{\sqrt{\sum_{i=1}^{n}\left(x_{i j}-\bar{x}_{j}\right)^{2}} \sqrt{\sum_{i=1}^{n}\left(x_{i k}-\bar{x}_{k}\right)^{2}}}
$$
is the Pearson correlation coefficient between variables $\mathbf{X}_{j}$ and $\mathbf{X}_{k}$

We can calculate the correlation matrix 

$$
\mathbf{R}=\frac{1}{n} \mathbf{X}_{s}^{\prime} \mathbf{X}_{s}
$$

where $\mathbf{X}_{s}=\mathbf{C X D}^{-1}$ with
  
- $\mathbf{C}=\mathbf{I}_{n}-n^{-1} \mathbf{1}_{n} \mathbf{1}_{n}^{\prime}$ denoting a centering matrix, 
- $\mathbf{D}=\operatorname{diag}\left(s_{1}, \ldots, s_{p}\right)$ denoting a diagonal scaling matrix.  

Note that the standardized matrix $\mathbf{X}_{s}$ has the form

$$
\mathbf{X}_{s}=\left(\begin{array}{cccc}
\left(x_{11}-\bar{x}_{1}\right) / s_{1} & \left(x_{12}-\bar{x}_{2}\right) / s_{2} & \cdots & \left(x_{1 p}-\bar{x}_{p}\right) / s_{p} \\
\left(x_{21}-\bar{x}_{1}\right) / s_{1} & \left(x_{22}-\bar{x}_{2}\right) / s_{2} & \cdots & \left(x_{2 p}-\bar{x}_{p}\right) / s_{p} \\
\left(x_{31}-\bar{x}_{1}\right) / s_{1} & \left(x_{32}-\bar{x}_{2}\right) / s_{2} & \cdots & \left(x_{3 p}-\bar{x}_{p}\right) / s_{p} \\
\vdots & \vdots & \ddots & \vdots \\
\left(x_{n 1}-\bar{x}_{1}\right) / s_{1} & \left(x_{n 2}-\bar{x}_{2}\right) / s_{2} & \cdots & \left(x_{n p}-\bar{x}_{p}\right) / s_{p}
\end{array}\right)
$$

How?


```r
# More direct
n <- nrow(X)
sdx <- 1/matrix(1, n, 1)%*%apply(X, 2, sd)
m <- matrix(1, n, 1)%*%colMeans(X)
Xs <- (X-m)*sdx
Xs
```

```
##               [,1]        [,2]        [,3]
##  [1,] -0.121717156 -0.81233989  1.66102560
##  [2,] -0.565704894  0.31273963 -1.23151117
##  [3,] -2.404843294  0.02690804 -0.77135887
##  [4,]  0.002581324 -1.21380031 -0.71032005
##  [5,] -0.126516525 -1.03412063  0.06726369
##  [6,]  0.656890910 -0.58746247  1.52435083
##  [7,]  0.510227259 -0.05725905 -0.55881729
##  [8,]  1.297945627  2.04462654 -0.01872963
##  [9,]  0.697496131  0.27445664  0.72022674
## [10,]  0.053640619  1.04625151 -0.68212986
```

```r
# Or
C <- diag(n) - matrix(1/n, n, n)
D <- diag(apply(X, 2, sd))
Xs <- C %*% X %*% solve(D)
Xs
```

```
##               [,1]        [,2]        [,3]
##  [1,] -0.121717156 -0.81233989  1.66102560
##  [2,] -0.565704894  0.31273963 -1.23151117
##  [3,] -2.404843294  0.02690804 -0.77135887
##  [4,]  0.002581324 -1.21380031 -0.71032005
##  [5,] -0.126516525 -1.03412063  0.06726369
##  [6,]  0.656890910 -0.58746247  1.52435083
##  [7,]  0.510227259 -0.05725905 -0.55881729
##  [8,]  1.297945627  2.04462654 -0.01872963
##  [9,]  0.697496131  0.27445664  0.72022674
## [10,]  0.053640619  1.04625151 -0.68212986
```

```r
# Or 
Xs <- scale(X, center=TRUE, scale=TRUE)

# Finally, the correlation Matrix
R <- t(Xs) %*% Xs / (n-1)
R
```

```
##           [,1]       [,2]       [,3]
## [1,] 1.0000000  0.2745780  0.3885349
## [2,] 0.2745780  1.0000000 -0.2644881
## [3,] 0.3885349 -0.2644881  1.0000000
```

```r
# Check it
cor(X)
```

```
##           [,1]       [,2]       [,3]
## [1,] 1.0000000  0.2745780  0.3885349
## [2,] 0.2745780  1.0000000 -0.2644881
## [3,] 0.3885349 -0.2644881  1.0000000
```

The correlations above are called "zero-order" or Pearson correlations.  They only reflect pairwise correlations without controlling other variables.  
  
## Precision Matrix  
  
The inverse of covariance matrix, if it exists, is called the concentration matrix also knows as the **precision matrix**.  

Let us consider a $2 \times 2$ covariance matrix:

$$
\left[\begin{array}{cc}
\sigma^{2}(x) & \rho \sigma(x) \sigma(y) \\
\rho \sigma(x) \sigma(y) & \sigma^{2}(y)
\end{array}\right]
$$

And, its inverse:

$$
\frac{1}{\sigma^{2}(x) \sigma^{2}(y)-\rho^{2} \sigma^{2}(x) \sigma^{2}(y)}\left[\begin{array}{cc}
\sigma^{2}(y) & -\rho \sigma(x) \sigma(y) \\
-\rho \sigma(x) \sigma(y) & \sigma^{2}(x)
\end{array}\right]
$$
  
If call the precision matrix $D$, the correlation coefficient will be   
  
$$
-\frac{d_{i j}}{\sqrt{d_{i i}} \sqrt{d_{j j}}},
$$
Or,
  
$$
\frac{-\rho \sigma_{x} \sigma_{y}}{\sigma_{x}^{2} \sigma_{y}^{2}\left(1-e^{2}\right)} \times \sqrt{\sigma_{x}^{2}\left(1-\rho^{2}\right)} \sqrt{\sigma_{y}^{2}\left(1-\rho^{2}\right)}=-\rho
$$

That was for a $2 \times 2$ variance-covariance matrix.  When we have more columns, the correlation coefficient reflects partial correlations. Here is an example:  
  

```r
pm <- solve(S) # precision matrix
pm
```

```
##               [,1]          [,2]          [,3]
## [1,]  0.0007662131 -0.0003723763 -0.0005985624
## [2,] -0.0003723763  0.0010036440  0.0005770819
## [3,] -0.0005985624  0.0005770819  0.0018907421
```

```r
# Partial correlation of 1,2
-pm[1,2]/(sqrt(pm[1,1])*sqrt(pm[2,2])) 
```

```
## [1] 0.4246365
```

```r
# Or
-cov2cor(solve(S))
```

```
##            [,1]       [,2]       [,3]
## [1,] -1.0000000  0.4246365  0.4973000
## [2,]  0.4246365 -1.0000000 -0.4189204
## [3,]  0.4973000 -0.4189204 -1.0000000
```

```r
# Or
ppcor::pcor(X)
```

```
## $estimate
##           [,1]       [,2]       [,3]
## [1,] 1.0000000  0.4246365  0.4973000
## [2,] 0.4246365  1.0000000 -0.4189204
## [3,] 0.4973000 -0.4189204  1.0000000
## 
## $p.value
##           [,1]      [,2]      [,3]
## [1,] 0.0000000 0.2546080 0.1731621
## [2,] 0.2546080 0.0000000 0.2617439
## [3,] 0.1731621 0.2617439 0.0000000
## 
## $statistic
##          [,1]      [,2]      [,3]
## [1,] 0.000000  1.240918  1.516557
## [2,] 1.240918  0.000000 -1.220629
## [3,] 1.516557 -1.220629  0.000000
## 
## $n
## [1] 10
## 
## $gp
## [1] 1
## 
## $method
## [1] "pearson"
```

## Semi-partial Correlation

With partial correlation, we find the correlation between $X$ and $Y$ after controlling for the effect of $Z$ on both $X$ and $Y$. If we want to hold $Z$ constant for just $X$ or just $Y$, we use a semipartial correlation.
  
While a partial correlation is computed between two residuals, a semipartial is computed between one residual and another variable. One interpretation of the semipartial is that the influence of a third variable is removed from one of two variables (hence, semipartial). This can be shown with the $R^2$ formulation.

Partial:

$$
r_{12.3}^{2}=\frac{R_{1.23}^{2}-R_{1.3}^{2}}{1-R_{1.3}^{2}}
$$
  
Semi-Partial:

$$
r_{1(2.3)}^{2}=R_{1.23}^{2}-R_{1.3}^{2}
$$


Let's see the difference between a slope coefficient, a semi-partial correlation, and a partial correlation by looking their definitions:  
  
**Partial:**    
  
$$
r_{12,3}=\frac{r_{12}-r_{13} r_{23}}{\sqrt{1-r_{12}^{2}} \sqrt{1-r_{23}^{2}}}
$$

**Regression:**

$$
X_{1}=b_{1}+b_{2} X_{2}+b_{2} X_{3}
$$
and  

$$
b_{2}=\frac{\sum X_{3}^{2} \sum X_{1} X_{2}-\sum X_{1} X_{3} \sum X_{2} X_{3}}{\sum X_{2}^{2} \sum X_{3}^{2}-\left(\sum X_{2} X_{3}\right)^{2}}
$$

With standardized variables:

$$
b_{2}=\frac{r_{12}-r_{13} r_{23}}{1-r_{23}^{2}}
$$
  
**Semi-partial (or "part") correlation:**
   
$$
r_{1(2.3)}=\frac{r_{1 2}-r_{1_{3}} r_{23}}{\sqrt{1-r_{23}^{2}}}
$$

The difference between the regression coefficient and the semi-partial coefficient is the square root in the denominator. Thus, the regression coefficient can exceed $|1.0|$; the correlation cannot.  In other words, semi-partial normalizes the coefficient between -1 and +1.
  
The function `spcor` can calculate the pairwise semi-partial correlations for each pair of variables given others. 



```r
ppcor::spcor(X)
```

```
## $estimate
##           [,1]       [,2]       [,3]
## [1,] 1.0000000  0.3912745  0.4781862
## [2,] 0.4095148  1.0000000 -0.4028191
## [3,] 0.4795907 -0.3860075  1.0000000
## 
## $p.value
##           [,1]      [,2]      [,3]
## [1,] 0.0000000 0.2977193 0.1929052
## [2,] 0.2737125 0.0000000 0.2824036
## [3,] 0.1914134 0.3048448 0.0000000
## 
## $statistic
##          [,1]      [,2]      [,3]
## [1,] 0.000000  1.124899  1.440535
## [2,] 1.187625  0.000000 -1.164408
## [3,] 1.446027 -1.107084  0.000000
## 
## $n
## [1] 10
## 
## $gp
## [1] 1
## 
## $method
## [1] "pearson"
```
  


  

