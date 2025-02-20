# CycleResearcher: Improving Automated Research via Automated Review

[![GitHub stars](https://img.shields.io/github/stars/zhu-minjun/Researcher)](https://github.com/username/CycleResearcher/stargazers) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)



  ![](img/method.png)

## 🎯 Introduction

CycleResearcher is a groundbreaking open-source project aimed at improving automated research through automated review. Our system comprises two core models: CycleResearcher and CycleReviewer, working in tandem to simulate the complete research and review cycle.

Our objectives are:
- 🤖 Automate academic research processes

- 📝 Provide high-quality research reviews

- 🔄 Establish research-review feedback loops

- 🚀 Accelerate scientific discovery

  

## 🚀 Getting Started

`pip install -e .`

### 📚 Tutorials and Demos

We have prepared comprehensive tutorials for both CycleResearcher and CycleReviewer to help users better understand and utilize these models. Our tutorials cover everything you need to get started and make the most of our model suite.

##### Available Tutorials
- [Tutorial 1:]() Getting Started with CycleResearcher 🚀
- [Tutorial 2:]() Understanding CycleReviewer 📝
- [Tutorial 3:]() Advanced Features🔥 Rejection Sampling with CycleResearcher and CycleReviewer [TODO]



### 🔄 Review-5K and Research-14K datasets

  ![](img/dataset.png)

| Dataset Name | Train Data | Test Data |                           HF Link                            |
| :----------: | :--------: | --------- | :----------------------------------------------------------: |
|  Review-5k   |   4,189    | 781       | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-8B) |
| Research-14K |   12,696   | 802       | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-70B) |



### 📊 Model Overview

#### CycleReviewer Model

|          Model Name           |                 Pre-training Language Model                  |                           HF Link                            |
| :---------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CycleReviewer-ML-Llama3.1-8B  | [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-8B) |
| CycleReviewer-ML-Llama3.1-70B | [Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-70B) |
|   CycleReviewer-ML-Pro-123B   | [Mistral-Large-2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Pro-123B) |

#### CycleResearcher Model

|       Model Name        |                 Pre-training Language Model                  |                           HF Link                            |
| :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CycleResearcher-ML-12B  | [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-12B) |
| CycleResearcher-ML-72B  | [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-72B) |
| CycleResearcher-ML-123B | [Mistral-Large-2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/CycleResearcher-ML-123B) |



## 📈 Model Evaluation

#### CycleReviewer

  <img src="img/cyclereviewer.png"  />

Comparison of automated models on generating peer review including various API-based methods and ours. CycleReviewer clearly outperforms both proprietary systems and human experts in peer review tasks. Specifically, it achieves a 48.77\% reduction in Proxy MSE and a 26.89\% reduction in Proxy MAE compared to human reviewers. With a decision accuracy of 74.24\%, our model demonstrates a significant lead over other closed-source systems.

#### CycleResearcher

![.img/cycleresearcher.png](/img/cycleresearcher.png)

CycleResearcher-12B achieves an average score of 5.36, approaching the 5.69 average scores for conference-accepted papers and surpassing AI Scientist's score of 4.31.

## 🔍 Model Detection

```
from cycleresearcher.detect import FastDetectGPT
detector = FastDetectGPT(
    model_name="meta-llama/Llama-3.1-8B",
    reference_model_name=None, 
    device="cuda"
)


text = "paper..."
result = detector.detect(text)
print(f"Criterion: {result['criterion']:.4f}")
print(f"Probability of being machine-generated: {result['probability']*100:.0f}%")

# BATCH
texts = ["Paper 1...", "Paper 2...", "Paper 3..."]
results = detector.detect_batch(texts)
```



## 📚 Citation

If you use CycleResearcher in your research, please cite our paper:

```bibtex
@inproceedings{
anonymous2024cycleresearcher,
title={CycleResearcher: Improving Automated Research via Automated Review},
author={Anonymous},
booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=bjcsVLoHYs},
note={under review}
}
```

## 📄 License

This code is licensed under the *Apache 2.0* License, and the model weight under the *CycleResearcher-License*, see the [LICENSE](LICENSE.md) file for details.



## 📮 Contact

- Submit an Issue
- Email us at: zhuminjun@westlake.edu.cn, wengsyx@gmail.com
