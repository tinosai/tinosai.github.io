---
layout: post
title: Canonical Correlation Analysis
output:
  md_document:
    variant: markdown_github
    preserve_yaml: true
#  html_document:
#    toc: true
#    toc_depth: 2
#    css: style.css
---

``` r
rm(list=ls())
```

## Canonical Correlation Analysis

This notebook shows a possible implementation of CCA on the `mfeat-fac`
and `mfeat-fou` set. The files are available
[here](https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/).

## 1. Import all the relevant libraries

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
plt.style.use("ggplot")
```

## 2. Import the data files

``` python
with open("mfeat-fac", "r") as file:
    view1 = file.read().split("\n")
view1 = [el.strip().split() for el in view1[:-1]]

with open("mfeat-fou", "r") as file:
    view2 = file.read().split("\n")
view2 = [el.strip().split() for el in view2[:-1]]

# cast the data to numpy arrays 
view1 = np.array(view1, dtype=float)
view2 = np.array(view2, dtype=float)

# standardize function
scale = lambda M :  (M - M.mean(axis=0).reshape(1,-1))/M.std(ddof=1,axis=0).reshape(1,-1)

# apply standardization to the two data sets
view1_scaled = scale(view1)
view2_scaled = scale(view2)
```

## 3. Calculate the Covariance Matrices *Σ*<sub>*X**X*</sub> , *Σ*<sub>*X**Z*</sub>, *Σ*<sub>*Z**Z*</sub>

``` python
# number of examples for each matrix 
n = view1.shape[0]

# calculation of the covariance matrices
sigma_xx = 1/n*(view1_scaled.T @ view1_scaled)
sigma_zz = 1/n*(view2_scaled.T @ view2_scaled)
sigma_xz = 1/n*(view1_scaled.T @ view2_scaled)
```

``` python
# Now we show the covariance matrices Sigma_XX and Sigma_ZZ
fig, ax = plt.subplots(nrows=1, ncols=2)
```

    ## <string>:1: MatplotlibDeprecationWarning: The resize_event function was deprecated in Matplotlib 3.6 and will be removed two minor releases later. Use callbacks.process('resize_event', ResizeEvent(...)) instead.

``` python
ax[0].imshow(sigma_xx, cmap='magma', interpolation=None)
ax[0].set_title("$\\Sigma_{XX}$")
ax[0].axis("off");
ax[1].imshow(sigma_zz, cmap='magma', interpolation=None)
ax[1].set_title("$\\Sigma_{ZZ}$")
ax[1].axis("off");
```

<img src="//Users/fortunatonucera/blog/tinosai.github.io/images/2022-10-10/unnamed-chunk-7-1.png" width="672" style="display: block; margin: auto;" />

``` python
# Now we plot the inter-set covariance matrix
plt.imshow(sigma_xz.T, cmap="magma", interpolation=None)
plt.axis("off");
plt.title("$\\Sigma_{XZ}$")
plt.colorbar(fraction=0.02, pad=0.02)
```

<img src="//Users/fortunatonucera/blog/tinosai.github.io/images/2022-10-10/unnamed-chunk-9-3.png" width="672" style="display: block; margin: auto;" />

## 4. Calculate the inverse-square-root covariance matrices

Recall that, if *Σ*<sub>*X**X*</sub> = *P**D**P*<sup>−1</sup> then
*Σ*<sub>*X**X*</sub><sup>−1/2</sup> = *P**D*<sup>−1/2</sup>*P*<sup>−1</sup>
(diagonalization). <br> Also, the exponentiation of the diagonal matrix
of the eigenvalues *D* is done *elementwise, and only applied to the
diagonal.* All the other elements are 0. <br>

Note that the eigenvectors in the eigenvector matrix returned by
`np.linalg.eig` are already normalized. Therefore the eigenvector matrix
is orthogonal, therefore the transpose of the matrix is equal to its
inverse, and we can write:
*Σ*<sub>*X**X*</sub><sup>−1/2</sup> = *P**D*<sup>−1/2</sup>*P*<sup>*T*</sup>

### 4.1 Calculate *Σ*<sub>*X**X*</sub><sup>−1/2</sup>

``` python
# np.linalg.eig returns, as first element, the vector of eigenvalues. As second element, the matrix of eigenvectors
sigma_xx_eigval , sigma_xx_eigvect = np.linalg.eig(sigma_xx)

# clip negative eigenvalues arising due to numerical imprecisions
sigma_xx_eigval[sigma_xx_eigval < 0] = 1e-18

# reorder eigenvalues. May not be sorted, in principle
order = np.argsort(sigma_xx_eigval)
sigma_xx_eigval , sigma_xx_eigvect = sigma_xx_eigval[order], sigma_xx_eigvect[:,order]

sigma_xx_12 = sigma_xx_eigvect @ np.diag(sigma_xx_eigval**(-0.5)) @ sigma_xx_eigvect.T
```

### 4.2 Calculate *Σ*<sub>*Z**Z*</sub><sup>−1/2</sup>

``` python
# np.linalg.eig returns, as first element, the vector of eigenvalues. As second element, the matrix of eigenvectors
sigma_zz_eigval, sigma_zz_eigvect  = np.linalg.eig(sigma_zz)

# clip negative eigenvalues arising due to numerical imprecisions
sigma_zz_eigval[sigma_zz_eigval < 0] = 1e-18

# reorder eigenvalues. May not be sorted, in principle
order = np.argsort(sigma_zz_eigval)
sigma_zz_eigval , sigma_zz_eigvect = sigma_zz_eigval[order], sigma_zz_eigvect[:,order]

sigma_zz_12 = sigma_zz_eigvect @ np.diag(sigma_zz_eigval**(-0.5)) @ sigma_zz_eigvect.T
```

## 5. Calculate the matrix W (a sort of cross-correlation)

``` python
W = sigma_xx_12 @ sigma_xz @ sigma_zz_12
```

## 6. Decompose the cross-correlation matrix through SVD

``` python
# mind that the right singular vector matrix V is returned as transpose by the SVD
U, S, Vh = np.linalg.svd(W)
V = Vh.T
```

Now the columns of *U* and *V* are *u*<sub>*m*</sub><sup>\*</sup> and
*v*<sub>*m*</sub><sup>\*</sup> we have in the notes.

## 7. Obtain the canonical loadings

The matrices of the canonical loadings *u*<sub>*m*</sub> and
*v*<sub>*m*</sub> can be obtained as follows

``` python
u = sigma_xx_12 @ U
v = sigma_zz_12 @ V

m = np.min(list(u.shape)+list(v.shape))

# remove trailing columns for u or v
u = u[:,:m]
v = v[:,:m]
```

## 8. Generate the correlation graph between *X**u*<sub>*m*</sub> and *Z**v*<sub>*m*</sub>

``` python
# calculate the correlation coefficient. np.corrcoeff works but I like this way better
# the original matrices are normalized, so we don't need to divide
corrGraph = np.mean((view1_scaled @ u) * (view2_scaled @ v), axis=0) 

# plot the correlation line
plt.plot(np.arange(len(corrGraph)), corrGraph, c="k", marker="o", markersize=3, lw=0.5)
plt.xlabel("index")
plt.ylabel("correlation")
plt.title("Correlation graph between $Xu_m$ and $Zv_m$")
```

<img src="//Users/fortunatonucera/blog/tinosai.github.io/images/2022-10-10/unnamed-chunk-15-5.png" width="672" style="display: block; margin: auto;" />

``` python
corrGraph[0]
```

    ## 0.9713479031494102

Note how this value matches `R`’s correlation coefficient value

## 9. Make a scatterplot of the canonical variate pair (like in the lab)

``` python
plt.scatter(-(view1_scaled @ u)[:,1], 
            -(view2_scaled @ v) [:,0],
           c = "k",
           alpha = 0.2)
plt.xlabel("First View, second component")
plt.ylabel("Second View, first component")
plt.title("Canonical Variates")
```

<img src="//Users/fortunatonucera/blog/tinosai.github.io/images/2022-10-10/unnamed-chunk-17-7.png" width="672" style="display: block; margin: auto;" />

### Notes:

-   The eigenvectors’ direction is arbitrary. I put a “-” sign in the
    plot in order to obtain the same graph as the lab. If you don’t, it
    will look mirrored. All depends on the definition of the canonical
    loadings and on the underlying SVD of the correlation matrix.
-   This implementation could be optimized by using `np.linalg.svd`
    instead of `np.linalg.eig`. These two functions give the same
    results when the underlying matrix is symmetric positive
    semi-definite (as all covariance matrices are by definition)

</p>
<p style="text-align:center;">
[Home](https://tinosai.github.io/)
</p>
