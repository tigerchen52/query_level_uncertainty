import os
os.environ['TRANSFORMERS_CACHE'] = "/data/parietal/store3/soda/lihu/hf_model/"
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

# we consider the last k tokens of an input prompt, assuming that a model has seen the entire query and is able
# to infer its knowledge gap. For instruction-aligned models, the system tokens after the query are regarded as the last k tokens.
# Note that we can add suffix tokens for goal, e.g. {query}? your answer is;  {query}? please provide your confidence score
# The performance is sensitive to this value
last_k_token = 6


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
score = ql_uncertainty.estimate(query, last_k_token)
print(f"[internal_confidence] score = {score}")


# internal confidence with in-context learning
ql_uncertainty = QLUncertainty(model, tokenizer, method='internal_confidence', target_tokens=[[7566], [2360]])
examples = [{'query':'the capital of China is Beijing', 'answer': 'Yes' }, {'query':'the capital of Spain is London', 'answer': 'No'}]
score = ql_uncertainty.estimate(query, last_k_token, examples)
print(f"[internal_confidence with examples] score = {score}")
