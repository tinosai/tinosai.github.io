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
<center>$$\begin{align*} \sigma=\frac{1}{1+e^{-x}} \end{align*}$$ <br />
The forward step follows the following equations: <br /></center>
<center>$$z^{[1]}=xW^{[1]}+b^{[1]}$$ <br /></center>
<center>$$a^{[1]}=\sigma(z^{[1]})$$ <br /></center>
<center>$$z^{[2]}=a^{[1]}W^{[2]}+b^{[2]}$$ <br /></center>
<center>$$a^{[2]}=\sigma(z^{[2]})$$ <br /></center>
<center>$$z^{[3]}=a^{[2]}W^{[3]}+b^{[3]}$$ <br /></center>
<center>$$y=\sigma(z^{[3]})$$ <br /></center>

with: <br />
$x \in \mathcal{R}^{(1,8)}$ <br />
$W^{[1]} \in \mathcal{R}^{(8,4)}$ <br />
$b^{[1]} \in \mathcal{R}^{(1,4)}$ <br />
$z^{[1]} \in \mathcal{R}^{(1,4)}$ <br />
$a^{[1]} \in \mathcal{R}^{(1,4)}$ <br />
$W^{[2]} \in \mathcal{R}^{(4,3)}$ <br />
$b^{[2]} \in \mathcal{R}^{(1,3)}$ <br />
$z^{[2]} \in \mathcal{R}^{(1,3)}$ <br />
$a^{[2]} \in \mathcal{R}^{(1,3)}$ <br />
$W^{[3]} \in \mathcal{R}^{(3,2)}$ <br />
$b^{[3]} \in \mathcal{R}^{(1,2)}$ <br />
$z^{[3]} \in \mathcal{R}^{(1,2)}$ <br />
$y \in \mathcal{R}^{(1,2)}$ <br />

Keep in mind that the vector-matrix multiplication gives, as outcome, another vector having as many column as the matrix. <br />

**Backpropagation**
Before proceeding with the training, we need to define a metric for the error and study the sigmoid a bit more. <br />
*Metric* <br />
The sigmoid has been defined above. However, during the
