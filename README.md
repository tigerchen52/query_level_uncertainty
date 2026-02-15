# query_level_uncertainty
[![Release](https://img.shields.io/pypi/v/pub-guard-llm?label=Release&style=flat-square)](https://pypi.org/project/query-level-uncertainty/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.15429-b31b1b.svg)](https://arxiv.org/abs/2506.09669)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the repo for our work "Query-Level Uncertainty in Large Language Models". Our proposed **Internal Confidence** is much faster than answer-level approaches (left figure). More importantly, it can be applied to adaptive inference, e.g., RAG, Deep Thinking, Cascading, and Abstention (right figure).

<p float="left">
  <img src="figure/qwen_gsm8k_pareto.png" width="40%" />
  <img src="figure/rag_running_time.png" width="40%" />
</p>

## Internal Confidence
The benefits of using our proposed internal confidence
* Training-free. No requirements for training samples.
* Fast. Estimating uncertainty using only a single forward pass of a given query without generating any tokens.

## Usage
Install our query-level uncertainty
```python
pip install query-level-uncertainty
```
Choose an uncertainty method to use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ql_uncertainty import QLUncertainty

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="cuda:0"
            )
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

query = "what is the capital of France"


# max probability
ql_uncertainty = QLUncertainty(model, tokenizer, method='max_prob')
score = ql_uncertainty.estimate(query)
print(f"[max_prob] score = {score}")

# predictive entropy
ql_uncertainty = QLUncertainty(model, tokenizer, method='pd_entropy')
score = ql_uncertainty.estimate(query)
print(f"[pd_entropy] score = {score}")

# Min-K Entropy
ql_uncertainty = QLUncertainty(model, tokenizer, method='mink_entropy')
score = ql_uncertainty.estimate(query)
print(f"[mink_entropy] score = {score}")


# Attentional Entropy 
ql_uncertainty = QLUncertainty(model, tokenizer, method='attn_entropy')
score = ql_uncertainty.estimate(query)
print(f"[attn_entropy] score = {score}")


# Perplexity 
ql_uncertainty = QLUncertainty(model, tokenizer, method='ppl')
score = ql_uncertainty.estimate(query)
print(f"[ppl] score = {score}")

# internal confidence. It is necessary to have token ids for Yes (7566) and No (2360). Repalce target tokens when using different LLMs
ql_uncertainty = QLUncertainty(model, tokenizer, method='internal_confidence', target_tokens=[[7566], [2360]])
score = ql_uncertainty.estimate(query)
print(f"[internal_confidence] score = {score}")


# internal confidence with in-context learning
ql_uncertainty = QLUncertainty(model, tokenizer, method='internal_confidence', target_tokens=[[7566], [2360]])
examples = [{'query':'the capital of China is Beijing', 'answer': 'Yes' }, {'query':'the capital of Spain is London', 'answer': 'No'}]
score = ql_uncertainty.estimate(query, examples)
print(f"[internal_confidence with examples] score = {score}")
```

## Reproduction!!
`eval_ql_uncertainty` folder includes the code for reproducing the key results in our paper

### Table 1

```python
python run_baseline.py
```
expected output example

```
microsoft/Phi-3-mini-4k-instruct max_prob
55.5&11.0&----&51.4&3.1&----&55.0&6.4&----&54.0&6.8&----\cr
microsoft/Phi-3-mini-4k-instruct pd_entropy
58.9&18.9&----&51.2&2.3&----&63.6&25.7&----&57.9&15.6&----\cr
```
Each line corresponds to a row in Table 1, indicating the performance of a particular baseline

```python
python run_ic.py --model_name Qwen/Qwen2.5-14B-Instruct
```
expected output example

```
+++++++++++++P(YES) TOP RIGHT+++++++++++++
Qwen/Qwen2.5-14B-Instruct
67.8&35.9&30.3&60.0&20.5&24.1&55.0&8.4&6.4&60.9&21.6&20.3\cr
+++++++++++++P(YES) NAIVE AVG+++++++++++++
Qwen/Qwen2.5-14B-Instruct
67.0&33.2&3.5&59.5&19.3&14.6&64.0&27.5&32.4&63.5&26.7&16.8\cr
+++++++++++++INTERNAL CONFIDENCE+++++++++++++
Qwen/Qwen2.5-14B-Instruct
71.9&44.8&26.5&62.6&24.6&18.2&66.8&34.6&5.7&67.1&34.7&16.8\cr
```

Note that the AUROC and ECE should be exactly the same, as reported in the Table, but there are some small fluctuations of PRR as it compares to a random baseline and this process introduces the randomness.

### Figure 6 (a)

```python
python run_rag.py
````
After, there will be a figure file in the same folder. Note that the optimal point might be changed due to the reproduction, but the overall trend is similar. 

In line 180, you can see `#confidence = random.random()`, which means using a random score to guide this rag process. 
This random score does show the same pattern as our internal confidence and the plateau disappears.



## Adaptive Inference
In terms of applications, we showcase that our proposed method can help efficient RAG and model cascading. 
On the one hand, internal confidence can guide users to assess the trade-offs between cost and quality when invoking additional services. On the other hand, it brings an ``optimal point'', where inference overhead can be reduced without compromising performance.

<p float="left">
  <img src="figure/rag_acc_cost.png" width="45%" />
  <img src="figure/cascade_acc_cost.png" width="45%" />
</p>

## Citation
If you find our work useful, give us a star or citation. Thank you!

