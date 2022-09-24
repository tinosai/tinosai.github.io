---
layout: post
title: Quadratic Approximation
output:
  md_document:
    variant: markdown_github
    preserve_yaml: true
#  html_document:
#    toc: true
#    toc_depth: 2
#    sansfont: Garamond
---

## It’s been a minute!

Hi! It’s been a long time since I last posted on this blog. There are
multiple reasons for that:

1.  Work has become more demanding, which means I have less time devote
    to writing.  
2.  I have started a M.Sc. program in Machine Learning and Data Science.

Going back to school is allowing me to partially fill the void in
statistics that many years of engineering have left behind. Even though
I was on the Neural Networks hype when I started this blog, I am
beginning to find Statistics (with the capital “S”) more and more
interesting. As a result, I have decided that I will focus on a broader
range of topics - which for the most part will be statistics-related.

Although I have tried publishing on Medium in the past, I feel that the
platform is not mature enough for anything related to math. The lack of
MathJax support was a big disappointment in my case.

Without further ado, let’s begin!

## Background

Many of the readers may have come across Bayesian inference in the past.
A bit of details on the procedure are explained here.

The Bayesian framework allows the statistician to incorporate prior
beliefs into the inference process. By *prior*, we mean the “knowledge”
we think we might have about something *before* we observe anything
related to it. Frequentists may call it *prejudice*, but Bayesians argue
that, no matter how fair and objective we think we are, there will
always be some sort of subjectivity in the analysis. Think about this:
when you propose a model and a *likelihood* function for a certain
model, you are already incorporating your prior belief of what that
model’s output data should look like. Isn’t that, already, a subjective
statement?

Bayesians accept this subjectivity, and use the Bayes’ theorem to update
their beliefs about the model’s parameters. A statement of the Bayes’
Theorem is included in Equation 1:

The denominator of Equation (1), the *evidence*, although dependent on
the particular form of chosen likelihood and prior distribution, is
**constant**. This denominator is itself the root of all the hardships
of Bayesian Statistics: as the parameter space grows in dimensions (1
dimension for each parameter implies a 20-dimensional space for 20
parameters, for example) calculating this high-dimensional integral
becomes extremely hard and computationally unfeasible. This is exactly
the reason why **Markov Chain Monte Carlo (MCMC)** (I have written an
article on Medium about it, you can find it
[here](https://medium.com/@tinonucera/bayesian-linear-regression-from-scratch-a-metropolis-hastings-implementation-63526857f191))
has gained popularity and is the **most reliable technique for Bayesian
inference, as long as you use it properly**.

Another approach, which is very simple and sometimes brings satisfactory
results, is the Quadratic Approximation. Which I will discuss below.

## Explanation

The Bernstein-Von Mises Theorem states that, under some regularity
conditions (which are detailed, for example,
[here](https://arxiv.org/pdf/1907.09611.pdf)) the posterior distribution
is asymptotically normal. Remember, this is an *asymptotic result* which
means it may or may not hold true, depending on how much data you have.

The Quadratic Approximation uses this asymptotic result to approximate
the posterior with a normal distribution. How is the probability density
function of this - possibly multivariate - normal distribution obtained?
Well, we know that a normal is fully identified by two parameters:

*θ* ∈ ℝ<sup>*n*</sup> is the mean and
*Σ*<sub>*θ*</sub> ∈ ℝ<sup>*n* × *n*</sup> is the covariance matrix
(which is positive semi-definite by definition). Approximating the
posterior with a multivariate normal means calculating these two
parameters *μ* and *Σ*.

Recall that, if *μ* is the mean of the multivariate normal distribution,
then this must also be the median and the **mode**. The fact that the
mean and the mode coincide is useful because the mode is the highest
value in the distribution. As a result, in order to find *μ*, we may use
any maximization technique we like (well, not really, but bear with me
for now).

Now, let’s look at the Bayes’ Theorem once again (in Equation 1). Since
the *evidence* is a constant, the maximum of the fraction on the right
hand side corresponds to maximizing the numerator only. We can then
write a proportional relationship: Or, in logarithmic terms:

We can then apply the Taylor’s Theorem on the log-posterior (I will
refer to it as logp from this moment on), and build a second-order
approximation:

Mind you: *θ* and *μ* are, in general, vectors, and they contain the
quantities we are trying to infer on. As we said before, we are
approximating the normal around its maximum *μ*. This means that the
gradient ∇logp(*μ*) must be zero. The Taylor’s approximation then
simplifies to: Next, observe that ∇ ⊗ ∇logp(*μ*) is the Hessian matrix
at the maximum. I have decided to make the outer product explicit to
avoid confusion with the laplacian operator. We can call this quantity
*H* to make it look more friendly.

Do you remember when I said we could use any optimization technique we
like? Well, I was lying. We can’t (or at least shouldn’t). If our
optimization algorithm uses an approximation of the Hessian matrix for
the iterations, we may use such approximation of the Hessian to
construct an estimator of the covariance matrix of the posterior
distribution. The Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) is
one of a larger family of algorithms which relies on the progressive
improvement of the estimate of the Hessian. But why is the Hessian so
important? because the Hessian is equivalent to the negative [*observed
Fisher Information*](https://en.wikipedia.org/wiki/Fisher_information).

In particular:

And also, asymptotically, we have that the covariance matrix evaluated
at the maximum (*Σ*<sub>*μ*</sub>) is the inverse of the observed Fisher
information:

Recalling that in our assumption the posterior is normal leads to the
following result:

And this is mostly the end of the theoretical explanation. The following
section will show how to code a quadratic approximation to draw samples
from the posterior of a Bayesian linear regression model. We use such a
simple model because the posterior is readily available and the
comparison becomes easy.

## Code

First of all, we need to import all the relevant libraries.

``` python
from sklearn.datasets import make_regression # for a dummy data set generation
import numpy as np
np.random.seed(42)
import scipy.stats as scio
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
```

Then we use the `make_regression` function to create a dummy data set
for regression.

``` python
X, y = make_regression(n_samples=100,n_features=2, bias=0.0)
```

A linear model can be defined through the following conditional
probability statement. where:

-   *y* ∈ ℝ<sup>*n*</sup> is the response vector
-   *X* ∈ ℝ<sup>*n* × *d*</sup> is the design matrix
-   *β* ∈ ℝ<sup>*d*</sup> is the parameter vector
-   *σ*<sup>2</sup> is a variance parameter (which sets the noise level)

In order to obtain an “easy” posterior, we use the conjugate prior on
*β* and *σ* (normal-inverse-gamma): where *V* = *λ*<sup>−1</sup>*I* The
choice of *λ* is entirely up to us to decide.

### Exact Posterior

In [this book](https://link.springer.com/book/10.1007/978-3-030-82808-0)
at chapter 8, proposition 8.1, it is shown that, for such model, the
posterior is: where: The derivation of the exact posterior is not the
main concern of this post, so the interested reader can refer to the
mentioned book for that. We just need the exact posterior for comparison
with the approximate one. First, we will place a prior on all the
parameters:

``` python
a, b = 1,1
lam = 0.01
V = 1/lam * np.eye(X.shape[1])
```

The parameters of the posterior will then be:

``` python
n = X.shape[0]
Vn = np.linalg.pinv(np.linalg.pinv(V)+X.T@X)
mn = Vn@X.T@y
an = a+n/2
bn = b+0.5*(np.dot(y,y)-np.dot(y,X@mn))
```

We then draw 10000 samples from the posterior of *σ*<sup>2</sup> and,
conditional on these values, draw the parameter *β* from the
multivariate normal. We then create a Pandas `DataFrame` with the beta
samples and produce a seaborn `jointplot`:

``` python
sigma2_post = 1/scio.gamma(a=an, scale=1/bn).rvs(10000)
beta_post = np.zeros((2,sigma2_post.shape[0]))
for i, sigma2_eval in enumerate(sigma2_post):
    beta_post[:,i] = scio.multivariate_normal(mean=mn, cov=sigma2_eval*Vn).rvs()

df_exact = pd.DataFrame()
df_exact["beta_1"] = beta_post[0,:]
df_exact["beta_2"] = beta_post[1,:]

graph1 = sns.jointplot( x = df_exact["beta_1"], 
                        y = df_exact["beta_2"], 
                        kind = 'hex', 
                        xlim = [87.3, 88.3], ylim = [73.6, 74.5])
graph1.set_axis_labels(xlabel="$\\beta_1$",ylabel="$\\beta_2$")
```

The picture below shows 10000 samples from the exact posterior for *β*.
<img src="/images/2022-09-23/unnamed-chunk-7-1.png" width="576" style="display: block; margin: auto;" />

### Approximate Posterior

For the approximate posterior, we need to write the logp (unnormalized
log-posterior) to be maximized. There are two things to consider at this
point:

-   Scipy’s optimization routine performs minimizations. We need a
    maximization. In order to turn a maximization problem into a
    minimization one, all we need to do is to change the sign of the
    objective function so that maximizing logp can be done by minimizing
     − logp.
-   *σ*<sup>2</sup> is constrained to be strictly larger than zero. In
    order to discourage the optimization routine from returning illegal
    values of *σ*<sup>2</sup>, we return an infinitely large negative
    log-posterior when *σ*<sup>2</sup> ≤ 0.

Consider that, if *σ*<sup>2</sup> needs to be strictly larger than zero,
the parameter space *σ*<sup>2</sup> is defined on is not compact. This
implies a violation of the Bernstein-Von Mises theorem. How gross this
violation is depends on the use we make of the results. Also, remember
that *σ*<sup>2</sup>’s posterior does not follow a normal distribution
but an inverse Gamma, although we knew very well in advance we would be
dealing with an approximation.

The objective function to minimize is included below.

``` python
def objective(pars,a=1, b=1, lam = 0.01):
    beta = np.array(pars[:2])
    sigma2 = pars[2]
    if pars[2] <=0:
        # returns infinite negative logp when sigma^2 <= 0
        return np.inf
    V = 1/lam * np.eye(X.shape[1])
    # logpdf of sigma
    logp  = scio.invgamma( a =1, scale = 1/b).logpdf(sigma2)
    # logpdf of beta
    logp += scio.multivariate_normal(mean = np.zeros(beta.shape), cov = sigma2*V).logpdf(beta)
    # log likelihood
    logp += scio.multivariate_normal(mean = np.squeeze(X@beta.reshape(-1,1)), 
                                     cov=sigma2*np.eye(X.shape[0])).logpdf(y)
    return -logp
```

We then call the optimization routine:

``` python
opt= minimize(objective, x0=np.array([0.5,0.5,0.5]),method="BFGS")
```

As we explained before, we use the popular `BFGS` for optimization, as
it naturally returns the inverse of the Hessian matrix which can then be
used for sampling the posterior from the approximation (recall that in
the previous section we explained that the Observed Fisher Information
and the Hessian of the negative log-posterior are the same thing). `x0`
is the array of initial guesses for the optimization. The first two
parameters are the two components of *β*, while the third is
*σ*<sup>2</sup>.

The output of the optimization, `opt`, is really all we need. We can now
sample from the approximate posterior. The results are included below.

``` python
params_approx = scio.multivariate_normal(mean=opt["x"], cov=opt["hess_inv"]).rvs(10000)
df_approx = pd.DataFrame({"beta_1":params_approx[:,0],
                          "beta_2":params_approx[:,1]})
graph2 = sns.jointplot( x = df_approx["beta_1"], 
                        y = df_approx["beta_2"], 
                        kind = 'hex', 
                        xlim = [87.3, 88.3], ylim = [73.6, 74.5])
graph2.set_axis_labels(xlabel="$\\beta_1$",ylabel="$\\beta_2$")
```

<img src="/images/2022-09-23/unnamed-chunk-11-3.png" width="576" style="display: block; margin: auto;" />
If we visualize the two bivariate normal distributions for *β* side by
side, they look almost exactly the same. ![side-by-side bivariate
posterior for beta](../images/compounded_pics.png "beta posterior")

## Conclusion

We showed how to code the quadratic approximation of the posterior
distribution. In addition, we explained that this approximation may or
may not be appropriate, depending on:

1.  the particular models and parameters in use
2.  how close the bulk of the posterior density is to the boundaries of
    the parameter space

This code can be easily improved on by adding a parser for the specific
distributions, so that the model can be fed, for example, via an
external plain text file.

Although the approximation should be working well with arbitrary complex
models, the normality of the posterior distribution is an asymptotic
result, which means that it must hold true when the number of data
points becomes arbitrarily large, but for small-data problems, Markov
Chain Monte Carlo could be a more suitable option.

Thanks for reading to the very end.
<p style="text-align:center;">
<img src="../images/raccs.png" width="400">
</p>
