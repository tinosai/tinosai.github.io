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

$y=xA$
