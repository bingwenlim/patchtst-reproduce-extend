**DSA5106 Project Proposal**

**Publication:** Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). [*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.*](https://arxiv.org/abs/2211.14730) *International Conference on Learning Representations (ICLR)*. 

**Project Type:** Algorithm Development and Empirical Evaluation

**1\. Introduction and Research Problem**

Long-term time series forecasting (LTSF) is critical in domains such as energy demand prediction, financial markets, and traffic flow forecasting. While Transformer architectures have dominated natural language processing, their application to LTSF has historically been hindered by several limitations. 

First, classical Transformers suffer from a quadratic computational complexity O(T2) with respect to sequence length, making them inefficient for long temporal horizons. Secondly, standard point-by-point tokenization often leads to overfitting and provides poor inductive bias for capturing local temporal semantics and multi-scale seasonal patterns.

The chosen paper, PatchTST, addresses these limitations through two key innovations:

1. **Patching:** Segmenting the time series into fixed-length patches (tokens) to retain local semantic information while drastically reducing the computational complexity of the attention mechanism.  
2. **Channel Independence:** Treating each variable as an independent univariate sequence processed by a shared Transformer backbone, reducing parameter explosion and improving generalization.

**Core Research Question**

We aim to investigate:

How does patch-based tokenization improve long-term forecasting efficiency and generalization, and can relaxing the strict channel-independence assumption further enhance performance on highly correlated multivariate datasets?

**2\. Background Study Required**

To thoroughly understand the technical content and reproduce PatchTST, we will review:

* **Transformer Architectures for Time Series:** Literature on self-attention mechanisms, positional encodings, and complexity bounds.

* **Baseline LTSF Models:** Reviewing the architectures of Informer, Autoformer, and standard DLinear models to understand the baseline comparisons used in the paper.

* **Self-Supervised Learning (SSL) in Time Series:** Background on masked reconstruction objectives, which PatchTST uses for its optional pretraining phase.

**3\. Plans for Reproduction and Extension**

**3.1 Reproduction Plan**

We will reproduce PatchTST from scratch in PyTorch. This involves implementing the patching mechanism, the Transformer encoder backbone, and the prediction head without relying on the authors' original training scripts. We will then run it on 2 standard benchmark datasets that were not featured in the paper (e.g. weather, electricity), and compare it against baseline models (DLinear and Autoformer) to confirm the relative performance reported in the original paper. 

**3.2 Dataset Extension**

The original PatchTST model relies heavily on a Channel-Independent (CI) architecture, meaning each variable in a multivariate time series is processed independently.  To test the limits of this CI assumption, we will extend to evaluate PatchTST on a dataset with strong inter-variable dependencies to test whether the channel-independence assumption limits performance in correlated settings.   Similarly, we will be comparing its performance against the 2 baseline models as well.

**3.3 Architectural Extension** 

We will extend the paper by addressing a key limitation: the strict channel-independence assumption. While channel independence is less susceptible to overfitting, it fails to capture explicit cross-variable dependencies (e.g., the relationship between temperature and humidity in weather forecasting). 

We propose an **Adaptive Cross-Channel Attention Module**. After the independent patches are processed by the primary Transformer encoder, the learned representations will be fed into this new module to learn variable dependencies before the final prediction head.

**4\. Evaluation of Result**

To evaluate the success of our reproduction and the effectiveness of our extensions, we will employ the following methodology:

We will evaluate our reproduction and extensions using standard time-series forecasting metrics: **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.  We will also do comparisons on the **Computation Cost** of training and prediction. 

Our evaluation pipeline will conduct a direct comparative analysis across the three configurations:

1. Open-source baselines (DLinear, Autoformer)  
2. Our reproduced base PatchTST (Channel-Independent)  
3. Our Extended PatchTST (with Cross-Channel Attention)

We will run these models across our chosen datasets (two standard benchmarks and one highly correlated dataset). 

By comparing the Extended PatchTST against the base model specifically on the highly correlated dataset, we will be able to objectively evaluate our hypothesis: whether explicitly modeling cross-channel dependencies at a later stage in the network improves forecasting accuracy in strongly correlated environments.  We will also be able to evaluate how these additional layers affect the performance on standard datasets.

**5\. Planned Division of Work**

| Member | Responsibilities | Report Contribution |
| :---- | :---- | :---- |
| Yang Yanting Lim Bing Wen | Code the base PatchTST architecture (patching mechanism, Transformer backbone, prediction head) from scratch in PyTorch.   Set up the coding environment (github, uv), import and configure the DLinear and Autoformer models for comparisons Write the primary training and validation loops and the scripts to monitor Computational Cost of Training and Prediction.  | Write the methodology section for reproduction, model architecture description, and baseline training setup. |
| Huang Wenhui Chu Shi Yuan | Source, clean, and preprocess 2 standard benchmark datasets that were not featured in the paper (e.g. weather, electricity) and 1 new highly correlated dataset (e.g., financial data).  Implement the evaluation scripts for Mean Squared Error (MSE), Mean Absolute Error (MAE). Run the models on these datasets.  | Write the dataset descriptions, preprocessing steps, evaluation metric definitions, and the baseline findings. |
| Wang Yuxiao Gao Ziming | Research, design, and code the Adaptive Cross-Channel Attention Module in PyTorch to create the new Extended PatchTST model. Train and evaluate this new extended model using the exact same datasets and evaluation pipelines  | Write the extension design, implementation details, comparative analysis results, and the final discussion. |
|  |  |  |

