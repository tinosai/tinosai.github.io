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

## Notation
$b \in \mathcal{R}$, therefore $b$ is a scalar.<br />
**x** $\in \mathcal{R}^N$ for $N>1$, therefore **x** is a vector.<br />
**W** $\in \mathcal{R}^{N,M}$ for $N,M>1$, therefore **W** is a matrix.<br />
Please note that **x** and **W** both use bold (meaning that they are tensors). In addition, lower-case letters will from this point on indicate vectors, whereas upper-case letters will denote matrices.<br />

Training a neural network consists of two steps: the forward step and the backpropagation. <br />
First, let's take a look at the picture below.
![picture](/assets/pictures/nn.001.jpeg)

The picture above shows a fully connected neural network with 2 hidden layers, an input layer with 8 units and an output layer with two units.
For simplicity, we will assume that the activation functions of each layer are of sigmoid type, where sigmoid $\sigma$ is: <br />
<center>$\begin{align*} \sigma=\frac{1}{1+e^{-x}} \end{align*}$ <br /></center>
The forward step follows the following equations: <br />
<center>$z^{[1]}=xW^{[1]}+b^{[1]}$ <br />
$a^{[1]}=\sigma(z^{[1]})$ <br />
$z^{[2]}=a^{[1]}W^{[2]}+b^{[2]}$ <br />
$a^{[2]}=\sigma(z^{[2]})$ <br />
$z^{[3]}=a^{[2]}W^{[3]}+b^{[3]}$ <br />
$y=\sigma(z^{[3]})$ <br /></center>

with: <br />
<center>$x \in \mathcal{R}^{(1,8)}$ <br />
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
$y \in \mathcal{R}^{(1,2)}$ <br /></center>

Keep in mind that the vector-matrix multiplication gives, as outcome, another vector having as many column as the matrix. <br />

## Backpropagation <br />
Before proceeding with the training, we need to define a loss function, calculate the derivative of the sigmoid and provide a two properties of matrix calculus which will be useful when calculating the derivatives. <br />
#### Loss Function
The most common loss function for a neural network are the MSE-Loss (Mean Squared Error) and Crossentropy-Loss. The former is generally used on regression problems, the latter is more commonly found in classification problems. <br />
In this article, we will consider a MSE-Loss, written as:<br />
<center>$E=\frac{1}{2m}\sum_{i=1}^m (g - y)$</center><br />
The single example leads to the following error: <br />
<center>$E=\frac{1}{2}(g_k - y_k)$</center><br />
Where g is the ground truth and y is the predicted value.

#### Derivative of the Sigmoid <br />
The sigmoid has been defined above. However, during the backpropagation step we will have to use its derivative, which I will show here: <br />
<center>$\frac{d}{dx}\sigma(x)=\frac{d}{dx}\frac{1}{1+e^{-x}}=\frac{e^{-x}}{(1+e^{-x})^2}=\sigma(x) \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \frac{1-1+e^{-x}}{1+e^{-x}}=\sigma(x) (1-\sigma(x))$</center><br />

#### Note on matrix calculus
It is hard to find resources explaining matrix calculus in detail. A couple months ago, I found this ebook that was quite clarifying: ["The matrix Cookbook"](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf). <br />
The hardest of backpropagation is deriving with respect to the weights. It is important to keep in mind that what I am about to state applies **only** to the denominator layout formulation of the neural network. If you use a numerator layout, you will need to work on the transpose to make the dimensions match. <br />
**The rules** <br />
Given a product **aX** where **a** is a vector and **X** is a matrix, its derivatives are: <br />
<center>$\frac{d}{da}(aX)=X$</center> <br />
<center>$\frac{d}{dX}(aX)=a^T$</center> <br />
Keep in mind that when the two rules above are applied to the chain rule in the neural network, $a^T$ has to be first term of the chain and $X$ has to be the last.


### The backpropagation process
We need to update the matrices $W^{[1]}$, $W^{[2]}$, $W^{[3]}$ as well as the vectors $b^{[1]}$,$b^{[2]}$,$b^{[3]}$. The optimization is performed with gradient descent, therefore we need to find the following gradients: <br />
<center>$\frac{dE}{dW^{[1]}},\frac{dE}{dW^{[2]}},\frac{dE}{dW^{[3]}},\frac{dE}{db^{[1]}},\frac{dE}{db^{[2]}},\frac{dE}{db^{[3]}}$</center> <br />
Now, I have re-written the neural network above explicitly showing the output of the layer ($z^{[k]}$) before the application of the activation function. <br />
Let's start from $\frac{\partial E}{\partial W^{[3]}}$. The application of the chain rule allows to write the derivative as a multiplication of simpler derivatives as stated below:
<center>\frac{\partial E}{\partial W^{[3]}}=\frac{\partial z^{[3]}}{\partial W^{[3]}}\frac{\partial y}{\partial z^{[3]}}\frac{\partial E}{\partial y}</center><br />
