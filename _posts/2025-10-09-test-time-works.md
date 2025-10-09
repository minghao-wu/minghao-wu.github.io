---
title: 'When Test-Time Scaling Works'
date: 2025-10-08
permalink: /posts/2025-10-08-test-time-works
tags:
  - test-time scaling
  - large language models
mathjax: true
---

Test-time scaling (TTS) involves investing more computational resources during inference to enhance model performance on complex tasks, especially reasoning. There are various TTS methods, such as chain-of-thought reasoning, self-consistency, and majority voting. As long as the method involves additional computation during inference, it can be considered a TTS method. 

Today, I encountered a machine translation paper claiming that TTS does not work for general MT but works for domain-specific MT ([Li et al., 2025](https://arxiv.org/abs/2510.06471)). This claim intrigued me, prompting me to reflect on when TTS methods are effective. 

Their findings align with my own experiences and intuitions. I think the relative effectiveness of TTS depends on two main factors: the model capability/performance and the task complexity/difficulty. As illustrated in the figure below, suppose the task complexity is properly defined. TTS methods are more likely to be effective when the model performance is moderate. Either when the model performance is very poor or very strong, TTS methods are less likely to be effective. Likewise, given a model with a certain capability, TTS methods are more likely to be effective when the task complexity is moderate. When the task complexity is very low or very high, TTS methods are less likely to be effective.

![Relative effectiveness of Test-Time Scaling](/images/posts/2025-10-09-test-time-works/test-time-scaling-2025-10-09-1104.svg)

If the intuition above holds, there are two possible directions for future research:

First, maybe we can use the effectiveness of TTS methods as a proxy to estimate the model capability and task complexity? For example, if TTS methods are effective on a certain task, it suggests that the model has moderate capability and we have not exhausted the potential of the model. If TTS methods are not effective, it could indicate that either the model is too weak or too strong for the task, or that the task is too easy or too hard.

Second, maybe we can find a metric from the model internal states (e.g., confidence, entropy, etc.) that predicts the model confidence in its predictions. If we can identify such a metric, we can potentially use it to determine when TTS methods would be beneficial. For instance, if the model exhibits high confidence in its predictions, it may indicate that TTS methods are unnecessary. Conversely, if the model shows low confidence, it may suggest that TTS methods could help improve performance. If such a metric can be found, it would be a significant step towards more efficient and effective use of TTS methods in practice and would help avoid the "overthinking" problem in large reasoning models.

![Claude is Overthinking in the divide calculation by Andrej Karpathy](images/posts/2025-10-09-test-time-works/overthinking.jpeg)

These thoughts are just my personal musings and may not be entirely accurate. I need to think more deeply about this topic and conduct more experiments to validate these hypotheses.