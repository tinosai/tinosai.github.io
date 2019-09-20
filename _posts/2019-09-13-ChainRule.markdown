---
layout: post
mathjax: true
comments: true
title:  "Chain"
date:   2019-09-08 08:37:06 +0900
categories: artificial intelligence update machine learning
---
{%- include mathjax.html -%}

Hello everyone! As I promised, I am publishing the first article about the chain rule for a fully connected neural network.

First, I will explain the basics of the *feedforward* step for a fully connected neural network. I will derive the vector formulation *backpropagation* and provide some pseudocode of the latter (up until last week I was considering to publish the python implementation but for backprop but for the sake of generality I will try to keep the code as much language agnostic as I can).

First of all, some math. In this blog I will use the denominator layout notation. It means that the vectors will be treated as **row vectors**. This affects what the equations look like and may cause some confusion at the beginning, but it won't be a big problem once the reader gets used to it.
Also, for the sake of readability, I will introduce some signs indicating whether a certain element is a scalar, a vector or a matrix.

**Notation**
$b \in \mathcal{R}$, therefore $b$ is a scalar.
\textbf{x} $\in \mathcal{R}^N$ for $N>1$, therefore $\textbf{x}$ is a vector.
\textbf{W} $\in \mathcal{R}^{N,M}$ for $N,M>1$, therefore \textbf{W}$ is a matrix.  
Please note that $\textbf{x}$ and $\textbf{W}$ both use bold (meaning that they are tensors). In addition, lower-case letters will from this point on indicate vectors, whereas upper-case letters will denote matrices.
