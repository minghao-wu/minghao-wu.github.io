---
title: 'Squeezing Your Fine-Tuning Data to the Last Drop: From Selection to Rebalancing'
date: 2025-11-14
permalink: /posts/2025-11-14-data-ilcc-talk
tags:
  - cool posts
  - category1
  - category2
mathjax: true
---

This post is based on the [talk](https://informatics.ed.ac.uk/ilcc/friday-14-november-11am) I give at the Institute for Language, Cognition and Computation (ILCC), University of Edinburgh, on November 14, 2025. The slides are available [here](/files/slides/2025-11-14-ILCC-talk.pdf).

## Abstract

The quality and composition of training data are paramount for the effective supervised fine-tuning (SFT) of large language models (LLMs). This talk presents two independent studies that tackle the challenge of data optimization from different, yet complementary, angles. The first study introduces GraphFilter, a novel data selection method that formulates the selection process as a set cover problem. By modeling the dataset as a bipartite graph and employing a priority function that balances quality and diversity, GraphFilter iteratively selects the most informative examples for training. The second study presents Mixture-of-Skills (MoS), a reinforcement learning framework designed to optimize data usage during fine-tuning. MoS dynamically adjusts the focus on different datasets to ensure balanced skill development in LLMs. Together, these two studies offer a comprehensive look at the data optimization landscape, providing valuable insights into both static data selection and dynamic data utilization for building more capable LLMs.

---

I recently had the privilege of presenting at the Institute for Language, Cognition and Computation (ILCC) at The University of Edinburgh, sharing insights from two complementary studies on data optimization for large language model fine-tuning. The talk, titled "Squeezing Your Fine-Tuning Data to the Last Drop: From Selection to Rebalancing," explored how we can systematically optimize training data to maximize model performance.


## Why Data Optimization Matters

When we fine-tune large language models, we're essentially teaching them new skills or refining existing ones. But here's the thing – not all training examples are created equal, and the way we use them during training can make or break the entire process.

Think about it this way: training data is the foundation upon which model capabilities are built. Every single example in your training set influences what your model learns. This creates three fundamental requirements that any serious fine-tuning effort must address:

**Quality** – We need examples that provide clear, correct, and informative training signals. High-quality data determines how well the model learns from each training instance.

**Diversity** – We need comprehensive coverage across all the skills and domains we want our model to master. Without diversity, we end up with models that excel in narrow areas but fail to generalize.

**Optimization** – We need efficient use of training resources throughout the learning process. This affects how effectively the model develops different capabilities over time.

These requirements translate into three concrete challenges: the **Selection Challenge** (what data should we train on?), the **Composition Challenge** (how should we mix different datasets?), and the **Optimization Challenge** (when should we use what data?). In the era of massive datasets containing millions of examples with wildly varying quality, we need principled methods beyond random sampling. Real-world fine-tuning typically involves multiple datasets covering different skills, and models don't learn all skills at the same rate – a model might quickly master conversational abilities but need thousands more steps to improve at mathematical reasoning.

## Our Data Optimization Pipeline

To address these challenges systematically, we developed a comprehensive data optimization pipeline:

**Large Dataset → <span style="color:blue;">Static Selection</span> (GraphFilter) → Selected Subset → <span style="color:red;">Dynamic</span> <span style="color:orange;">Rebalancing</span> (Mixture-of-Skills) → Optimized Training**

In this pipeline, we tackle data optimization from two complementary angles. **Before training**, we use static selection to curate the best possible training data, balancing both quality and diversity in our initial dataset selection. **During training**, we use dynamic rebalancing to adaptively adjust how we use this curated data, responding to the model's evolving learning needs throughout the fine-tuning process.

After extensive research, my collaborators and I developed a comprehensive framework that addresses these challenges through two complementary approaches:

**Stage 1: Static Selection (GraphFilter)** – Before training begins, we systematically select optimal data subsets that balance both quality and diversity. This isn't about choosing either high-quality examples OR diverse examples – it's about finding examples that excel in both dimensions simultaneously.

**Stage 2: Dynamic Rebalancing (Mixture-of-Skills)** – During training, we adaptively adjust how we sample from different datasets based on the model's evolving learning state. As the model masters some skills and struggles with others, our system automatically rebalances to focus resources where they're needed most.

What makes this approach particularly powerful is that these two stages are truly complementary. GraphFilter optimizes "what to train on" while Mixture-of-Skills optimizes "how to train." Together, they provide systematic optimization across the entire fine-tuning process.

## Study 1: GraphFilter - The Best of Both Worlds

Our first study, "[The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](https://www.arxiv.org/abs/2410.12458)," was accepted at ICML 2025. This work tackles one of the most persistent challenges in machine learning: how do you select training data that is both high-quality AND diverse?

### The Quality-Diversity Dilemma

For years, data selection methods have been trapped in a false binary choice. Quality-focused approaches select only the highest-scoring examples based on metrics like perplexity or reward scores. These methods identify excellent individual samples but often select very similar examples that don't provide comprehensive coverage of skills and domains.

On the flip side, diversity-focused methods maximize coverage across different topics, domains, or linguistic patterns. They ensure broad training coverage but often sacrifice quality by including lower-quality examples just to fill coverage gaps.

This creates what I call the "library curation problem." Imagine you're building a university library with limited shelf space. You could choose only the highest-rated books (excellent but narrow) or ensure coverage of every subject (comprehensive but mixed quality). What you really want is both excellent books AND comprehensive coverage.

GraphFilter's key insight is that this doesn't have to be an either-or choice. We can systematically find examples that excel in both dimensions simultaneously.

### From Set Cover to Data Selection

The theoretical foundation of GraphFilter lies in the classic set cover problem from computer science. Given a universe of elements and a collection of sets, the goal is to find the minimum number of sets that together cover all elements in the universe.

Here's how we transform this into data selection: each training sentence becomes a "set" containing various linguistic features (n-grams) as "elements." The goal becomes selecting the minimum number of sentences that together cover all desired linguistic patterns. This directly captures our dual objectives – comprehensive coverage with minimal redundancy.

*[Image placeholder: Diagram showing the set cover formulation with sentences as sets and n-grams as elements]*

### The Bipartite Graph Representation

GraphFilter implements this set cover formulation through an elegant bipartite graph structure. On one side, we have nodes representing training sentences. On the other side, we have nodes representing n-grams or linguistic features. Edges connect sentences to their contained features.

This representation makes coverage relationships explicit and enables efficient computation. When we select a sentence, we can immediately see which n-grams it covers and update the graph accordingly.

*[Image placeholder: Bipartite graph visualization showing sentences connected to their n-grams]*

But here's the crucial innovation: we don't just greedily select based on coverage. Instead, we use a multiplicative priority function that combines both quality and diversity:

**φ(u) = Quality(u) × Diversity(u)**

For quality, we use the Instruction-Following Difficulty (IFD) score, which measures how much more informative an instruction-response pair is compared to the response alone. For diversity, we sum the TF-IDF scores of all n-grams in the sentence.

The multiplication is key – it ensures that both quality AND diversity must be high for a sentence to achieve high priority. This eliminates the traditional trade-off between these dimensions.

### The Algorithm in Action

GraphFilter operates through an iterative selection process:

1. **Initialize** with an empty selection
2. **Compute** priority scores using our quality × diversity function  
3. **Select** the sentence with highest priority
4. **Update** by removing the selected sentence and its covered n-grams
5. **Repeat** until reaching the target budget

As we select sentences and remove covered n-grams, the priorities of remaining sentences automatically adjust. A initially low-priority sentence might become highly attractive if it covers rare, unselected patterns that complement our existing selection.

<p>
    <img src="/images/posts/2025-11-14-ilcc/graphfilter_example.png" alt>
    <em>"An illustrative example of GraphFilter in action, showing how sentence priorities evolve as n-grams are covered."</em>
</p>

### Key Insights and Design Choices

You can find the full details of our experiments and results in the ICML 2025 paper, but here are some of the key insights we uncovered:

**N-gram combinations matter**: Using unigrams + bigrams + trigrams significantly outperforms individual n-gram types, with each level providing complementary information about vocabulary, local relationships, and phrasal patterns.

**Trigrams hit the sweet spot**: Performance peaks at trigrams (n=3), balancing meaningful linguistic patterns with computational efficiency. Beyond trigrams, we see diminishing returns despite increased computational cost.

**Instruction diversity is crucial**: When working with instruction-response pairs, focusing GraphFilter on instruction diversity proves more impactful than response diversity, though both contribute to final performance.

**Budget affects strategy**: With small budgets (1K-5K examples), quality-focused methods excel. With larger budgets (10K+), diversity becomes more valuable. GraphFilter consistently delivers across all budget levels.


### Computational Efficiency

Beyond performance, GraphFilter delivers practical efficiency. Using max-heap operations and localized priority updates, it selects 10K instances from 300K candidates in just 2.48 hours – dramatically faster than methods requiring multiple model evaluations.

This efficiency makes GraphFilter viable for production use, not just academic research. Teams can experiment with different selection budgets and quality metrics without prohibitive computational overhead.

## The Next Challenge: Dynamic Optimization

GraphFilter demonstrated the power of balancing quality and diversity in static data selection. But it raised an important question: what about optimization during training? Models don't learn all skills at the same pace – they might quickly master conversational abilities but struggle with mathematical reasoning for thousands more steps.

This insight led us to our second study: how can we dynamically optimize data usage throughout the fine-tuning process? That's where Mixture-of-Skills comes in, but that's a story for the next section of our data optimization journey.

## Study 2: Mixture-of-Skills - Learning to Optimize Data Usage

Our second study, "[Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models](https://arxiv.org/abs/2406.08811)," published at EMNLP 2024, tackles the "Optimization Challenge."

While GraphFilter helps us decide *what* to put in our training bucket, it doesn't tell us *how* to spoon it out during training. Traditional fine-tuning uses static mixing—if you have 20% math data and 80% conversation data, that ratio stays fixed from step 1 to step 10,000.

**The Reality of Data Mixing**

Real-world datasets are rarely perfectly balanced. We often face a situation where we have 10K math samples, 100K coding samples, and 1M conversation samples. Current approaches to handling this heterogeneity have critical limitations:

*   **Static Mixing**: Treats all datasets equally throughout training, ignoring actual learning dynamics.
*   **Data Capping**: Limits large datasets to match smaller ones, discarding valuable examples.
*   **Grid Search**: Trying to find optimal mixing ratios manually is prohibitively expensive.

But learning is dynamic. A model might master simple conversational patterns early on but struggle with complex reasoning until much later. Static mixing ignores these evolving needs, potentially wasting resources on mastered skills while under-training difficult ones.

### The Solution: Bilevel Optimization with RL

Mixture-of-Skills (MoS) treats data mixing as a bilevel optimization problem:

1.  **Outer Level**: We optimize the LLM parameters to minimize loss (standard training).
2.  **Inner Level**: We optimize a "scorer network" that determines data sampling probabilities to maximize learning progress.

Since we can't easily differentiate through the discrete sampling process, we use Reinforcement Learning (specifically the REINFORCE algorithm). The scorer network acts as an agent, choosing which dataset to sample from, and receiving "rewards" based on how well that choice improved the model.

### Guiding the Learning: Three Reward Perspectives

To make this work, we need to define what "improvement" looks like. We designed three complementary reward functions:

*   **Transferability**: Measures how well learning from one dataset helps with others (using gradient similarity). This encourages cross-domain knowledge sharing.
*   **Difficulty**: Uses relative perplexity to identify where the model is struggling. This focuses attention on harder samples.
*   **Learning Trajectory**: Uses exponential moving averages to ensure stability, preventing the model from chasing noisy signals.

### Dynamic Adaptation in Action

The results were fascinating. When we visualized the sampling probabilities over time, we didn't see flat lines. Instead, we saw dynamic curves reflecting the model's learning phases.

<p>
    <img src="/images/posts/2025-11-14-ilcc/mos_sampling_probs.png" alt>
    <em>"MoS automatically adjusts sampling probabilities during training. Note how focus shifts to more challenging datasets (like Math) as training progresses."</em>
</p>

MoS consistently outperformed static baselines across various models (Qwen, Gemma, Llama-3). Crucially, it proved to be **complementary to data selection**. When we combined GraphFilter (to select high-quality data) with MoS (to optimize its usage), we achieved better performance than either method alone.

### MoSpec: Specialization through Dynamic Mixing

We extended this framework to a variant called **MoSpec**, designed for creating specialized models (e.g., a Math specialist).

A common intuition is that to build a math specialist, you should train only on math data. Our results challenged this. MoSpec showed that by dynamically mixing in "supporting" datasets (like logic or general conversation) alongside the target math data, we achieved significantly better performance on math benchmarks than training on math data alone (+4.87 points on GSM8K). It turns out that related skills provide necessary scaffolding for specialized capabilities.

## Conclusion: A Comprehensive Pipeline

Data optimization isn't a single step; it's a pipeline.

1.  **Static Selection (GraphFilter)** ensures we start with a dataset that is both high-quality and diverse, removing redundancy before we spend a single compute cycle on training.
2.  **Dynamic Rebalancing (Mixture-of-Skills)** ensures we use that data efficiently, adapting to the model's changing needs as it learns.

By addressing both "what to train on" and "how to train," we can squeeze every drop of performance out of our fine-tuning data.



