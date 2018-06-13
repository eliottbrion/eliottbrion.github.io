---
layout: default
title:  "Understanding the GAN value function"
author: "Chester"
---

I recently gained interest for Generative Adversarial Networks. Fascinated by both the theoritical idea and the various applications, I began the lecture of the orginal paper of Goodfellow et al. My enthousiams came down as I got stuch at the first equation. This equation describes the minimax problem that a GAN solves:

$$ \min_G \max_D V(D,G) = \mathbb E_{x \sim p_{data}(x)} [\log D(x)] + \mathbb E _{z \sim p_z (z)} [ \log(1-D(G(z))]$$ 

Thanks for checking out Tale!
