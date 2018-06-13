---
layout: default
title:  "Understanding the GAN value function"
author: "Chester"
---

I recently gained interest for Generative Adversarial Networks. Fascinated by both the theoritical idea and the various applications, I began the lecture of the orginal paper of Goodfellow et al., before being stuck in the first equation. This equation describes the minimax problem that a GAN solves:

$$ \min_G \max_D V(D,G) = E_{x \sim p_{data}(x)} [\log D(x)] \mathbb{E}$$ 

Thanks for checking out Tale!
