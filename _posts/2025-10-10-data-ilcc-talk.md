---
title: 'Squeezing Your Fine-Tuning Data to the Last Drop: From Selection to Rebalancing'
date: 2025-10-10
permalink: /posts/2025-10-10-data-ilcc-talk
tags:
  - cool posts
  - category1
  - category2
mathjax: true
---

This post is based on the [talk](https://informatics.ed.ac.uk/ilcc/friday-14-november-11am) I gave at the Institute for Language, Cognition and Computation (ILCC), University of Edinburgh, on November 14, 2025. The slides are available [here](/files/slides/2025-11-14-ILCC-talk.pdf).

## Abstract

The quality and composition of training data are paramount for the effective supervised fine-tuning (SFT) of large language models (LLMs). This talk presents two independent studies that tackle the challenge of data optimization from different, yet complementary, angles. The first study introduces GraphFilter, a novel data selection method that formulates the selection process as a set cover problem. By modeling the dataset as a bipartite graph and employing a priority function that balances quality and diversity, GraphFilter iteratively selects the most informative examples for training. The second study presents Mixture-of-Skills (MoS), a reinforcement learning framework designed to optimize data usage during fine-tuning. MoS dynamically adjusts the focus on different datasets to ensure balanced skill development in LLMs. Together, these two studies offer a comprehensive look at the data optimization landscape, providing valuable insights into both static data selection and dynamic data utilization for building more capable LLMs.

---

Coming soon...