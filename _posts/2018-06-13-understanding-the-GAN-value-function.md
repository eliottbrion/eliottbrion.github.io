---
layout: default
title:  "Understanding the GAN cost function"
author: "Eliott Brion"
---

# Understanding the GAN cost function



I recently gained interest in Generative Adversarial Networks (GANs). Fascinated by both the theoritical idea and the various applications, I dove into the original [paper](https://arxiv.org/abs/1406.2661) of Goodfellow et al. My enthusiasm quickly calmed down as I got stuck at the first equation. This equation is supposed to describe the minimax problem that a GAN aims to solve:

$$ \min_G \max_D V(D,G) = \mathbb E_{x \sim p_{data}(x)} [\log D(x)] + \mathbb E _{z \sim p_z (z)} [ \log(1-D(G(z))]$$ 

Despite my search on the web, I haven't found any explanation that is both simple and convincing. In this article, I propose to share how a few colleagues and I understand this equation now.

![Placeholder image](https://raw.githubusercontent.com/eliottbrion/eliottbrion.github.io/master/assets/GAN.PNG)

Intuitively, the goal of a GAN is to artificially generate images that resemble real-life images. Mathematically, we will define $p_{data}$ as the probability distribution of all the real-life images. Think of a robot that would travel the world and take photos randomly everywhere it goes. The process of taking all these photos randomly from the real world can be described by a distribution $p_{data}$. The goal of a GAN is to determine a distribution $p_g$ close to $p_{data}$, without having a robot traveling the world to take random photos of it. The image generator is $G$ (a neural network). It takes as input a sample from a noise distribution $p_z$ and returns an image $x=G(z)$ (i.e., $x ~\sim p_g$) that are real-life like. A second neural network $D$ takes as input both real images $x \sim p_{data}$ and fake images $x \sim p_g$ and is trained to distinguish between the two. If $Y$ is a random variable taking the value 1 if the image is real and 0 if the image is fake, the discriminator takes an image $x$ and outputs 

$$D(x)=P(Y=1 | x) $$

When training a GAN, we have samples $(x_i, y_i)$ ($i=1, 2, ..., 2m$) of images $x$ with their corresponding label ($y_i=1$ if $x_i$ has been drawn from $p_{data}$ and $y_i=0$ if $x_i$ has been drawn from $p_g$). Moreover, there are the same number of true and generated images ($m$ of each). Among all possible probability functions, we want to select the one that has the highest probability of having generated the labels that we observe

$$D = \arg \max_D \prod_{i=1}^{2m} P(y=y^{(i)} | x^{(i)}; D)$$

This is the maximum likelihood estimator. Since there are only two possible labels (for all $i$, $y^{(i)}$ is equal to either 1 or 0), this can be written as

$$D = \arg \max_D \prod_{i:y^{(i)}=1} P(y=1 | x^{(i)}; D) \prod_{i:y^{(i)}=0} P(y=0 | x^{(i)};D)$$

By definition, any function $x \mapsto D(x)$ outputs the probability of $x$ being a "true" image. Since an image is considered being either true or fake (there is no alternative), the probability $x$ being a "fake" image is $1-D(x)$. As a consequence,

$$D = \arg \max_D \prod_{i:y^{(i)}=1} D(x^{(i)}) \prod_{i:y^{(i)}=0} 1- D(x^{(i)})$$

Now since the function $x \mapsto \log x$ is a monotonically increasing function, solving this problem is equivalent to 

$$D = \arg \max_D \log \left ( \prod_{i:y^{(i)}=1} D(x^{(i)}) \prod_{i:y^{(i)}=0} 1- D(x^{(i)}) \right )$$

Writing the logarithm of the product as the sum of the logarithms yields

$$D = \arg \max_D  \sum_{i:y^{(i)}=1} \log D(x^{(i)}) + \sum_{i:y^{(i)}=0} \log( 1- D(x^{(i)})) $$

This expression does not change is we multiply the quantity to be maximized by any arbitrary number. Let's choose $1/m$

$$D = \arg \max_D \frac{1}{m} \sum_{i:y^{(i)}=1} \log D(x^{(i)}) + \frac{1}{m} \sum_{i:y^{(i)}=0} \log( 1- D(x^{(i)})) $$

For all generated images (for all $x^{(i)}$ such that $y^{(i)}=0$), there exists a $z^{(i)}$ that has been drawn from $p_z$ such that $x_i = G(z^{(i)})$

$$D = \arg \max_D \frac{1}{m} \sum_{i:y^{(i)}=1} \log D(x^{(i)}) + \frac{1}{m} \sum_{i:y^{(i)}=0} \log( 1- D(G(z^{(i)}))) $$

Intuitively, the law of large numbers (LLN) tells that $\frac{1}{m} \sum_{i=1}^m \log D(x^{(i)})$ gets arbitrarily close to $\mathbb{E}_ {x \sim p_{data}(x)} [\log D(x)]$ as $m$ increases to infinity. We say that the left quantity *converges in probability* (weak version of the LLN) or *converges almost surely* (strong version of the LLN) to the right quantity

$$\frac{1}{m} \sum_{i=1}^m \log D(x^{(i)}) \xrightarrow{\text{$m \rightarrow \infty$}} \mathbb E_{x \sim p_{data}(x)} [\log D(x)]$$

Similarly,

$$ \frac{1}{m} \sum_{i=1}^m \log( 1- D(G(z^{(i)}))) \xrightarrow{\text{$m \rightarrow \infty$}} \mathbb{E}_ {z \sim p_z(z)} [\log( 1- D(G(z)))]$$

This means that the distriminator aims to maximize the log-likelihood with an infinite number of samples

$$ D = \arg \max_D \mathbb E_{x \sim p_{data}(x)} [\log D(x)] +  \mathbb{E} _{z \sim p_z (z)} [ \log(1-D(G(z))]$$

The generator, on the contrary, aims at fooling the distriminator. In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G,D)$

$$ \min_G \max_D V(D,G) = \mathbb E_{x \sim p_{data}(x)} [\log D(x)] + \ \mathbb{E} _{z \sim p_z (z)} [ \log(1-D(G(z))] $$ 

And we are done.

Hope that this helps you get started with the understanding of GANs ;)




