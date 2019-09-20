---
layout: post
mathjax: true
comments: true
title:  "Chain Rule for Fully Connected Neural Network"
date:   2019-09-08 08:37:06 +0900
categories: artificial intelligence update machine learning
---
{%- include mathjax.html -%}

Hello everyone! As I promised, I am publishing the first article about the chain rule for a fully connected neural network.

First, I will explain the basics of the *feedforward* step for a fully connected neural network. I will derive the vector formulation *backpropagation* and provide some pseudocode of the latter (up until last week I was considering to publish the python implementation but for backprop but for the sake of generality I will try to keep the code as much language agnostic as I can).  <br />

First of all, some math. In this blog I will use the denominator layout notation. It means that the vectors will be treated as **row vectors**. This affects what the equations look like and may cause some confusion at the beginning, but it won't be a big problem once the reader gets used to it.  <br />
Also, for the sake of readability, I will introduce some signs indicating whether a certain element is a scalar, a vector or a matrix. <br />  

**Notation**  
$b \in \mathcal{R}$, therefore $b$ is a scalar.<br />
**x** $\in \mathcal{R}^N$ for $N>1$, therefore **x** is a vector.<br />
**W** $\in \mathcal{R}^{N,M}$ for $N,M>1$, therefore **W** is a matrix.<br />
Please note that **x** and **W** both use bold (meaning that they are tensors). In addition, lower-case letters will from this point on indicate vectors, whereas upper-case letters will denote matrices.<br />

Training a neural network consists of two steps: the forward step and the backpropagation. <br />
First, let's take a look at the picture below.
![picture](/assets/pictures/nn.001.jpeg)

The picture above shows a fully connected neural network with 2 hidden layers, an input layer with 8 units and an output layer with two units.
For simplicity, we will assume that the activation functions of each layer are of sigmoid type, where sigmoid $\sigma$ is: <br />
$$\sigma=\frac{1}{1+\exp(-x)}$$
The forward step follows the following equations:
$$z^{[1]}=xW^{[1]}+b^{[1]}$$
