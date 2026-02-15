import json
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MulticlassCalibrationError
import torch
import numpy as np
import itertools
import collections
from datasets import load_dataset

def compute_ece(y_true, y_prob, n_bins=10):
    
    y_prob = [[1-prob, prob] for prob in y_prob]
    assert len(y_true) == len(y_prob)
    preds = torch.tensor(y_prob)
    targets = torch.tensor(y_true)
    ece = MulticlassCalibrationError(num_classes=2, n_bins=n_bins, norm='l1')
    result = ece(preds, targets)
    return result.item()


def compute_prr(y_true, uncertainty_scores, polarity=1):
    y_true = np.array(y_true)
    auc_unc = roc_auc_score(y_true, polarity*uncertainty_scores) 

    random_scores = np.random.rand(len(y_true))
    auc_rnd = roc_auc_score(y_true, -random_scores)
    
    oracle_uncertainty = np.abs(1 - y_true) 
    auc_oracle = roc_auc_score(y_true, -oracle_uncertainty)

    prr = (auc_unc - auc_rnd) / (auc_oracle - auc_rnd)
    return prr

conf_methods = {
    'max_prob':0,
    'pd_entropy':0,
    'mink_entropy':0,
    'attn_entropy':0,
    'ppl':0,
}

model_names = {
    'microsoft/Phi-3-mini-4k-instruct': 3,
    'meta-llama/Llama-3.1-8B-Instruct': 6,
    'Qwen/Qwen2.5-14B-Instruct':6
}

dataset_names = {
    'trivia_qa': 1000,
    'sciq': 1000,
    'gsm8k': 1000,
}



def compare_calibration(model_name, dataset_name, estimator_name, dev_sample=1000):
    
    confidence_scores, labels = [], []
    model_name = model_name.split('/')[-1].lower()
    label_file_name = f"label/updated_{dataset_name}_{model_name}.json"
    dataset = load_dataset(
        "Lihuchen/query-level-uncertainty",
        data_files=label_file_name,
        split="train",
    )
    
    labels = []
    for index, obj in enumerate(dataset):
        if index >=10000:break
        labels.append(obj['label']) 
    
    base_score_path = f"baseline_data/{estimator_name}/"
    confidence_file_name = base_score_path + f"{dataset_name}_{estimator_name}_{model_name.split('/')[-1].lower()}.json"
    
    for line in open(confidence_file_name, encoding='utf8'):
        obj = json.loads(line)
        confidence = obj['confidence']
        confidence_scores.append(confidence)
        
    test_labels, dev_labels = labels[:-dev_sample], labels[-dev_sample:]
    test_confidence_scores, dev_confidence_scores = confidence_scores[:-dev_sample], confidence_scores[-dev_sample:]
    
    
    auc_score = roc_auc_score(test_labels[:len(test_confidence_scores)], test_confidence_scores)
    prr = compute_prr(test_labels[:len(test_confidence_scores)], test_confidence_scores, polarity=1)
    if estimator_name == 'semantic_similarity':
        ece = compute_ece(test_labels, test_confidence_scores)
    else: ece = 1000
    
    return round(auc_score*100, 1), round(prr*100, 1), round(ece*100, 1)

results = collections.OrderedDict()
combinations = list(itertools.product(conf_methods.keys(), model_names.keys(), dataset_names.keys()))
for conf_method, model_name, dataset_name in combinations:
    print(conf_method, model_name, dataset_name)
    auc_score, prr, ece = compare_calibration(model_name, dataset_name, conf_method, dev_sample=dataset_names[dataset_name])
    
    if model_name not in results:results[model_name] = collections.OrderedDict()
    if conf_method not in results[model_name]: results[model_name][conf_method] = list()
    results[model_name][conf_method].extend([auc_score, prr, ece])
    
    
for conf_method, values in results.items():
    for model, metrics in values.items():
        print(conf_method, model)
        auc = sum([metrics[i] for i in range(0, len(metrics)-2, 3)]) / 3
        prr = sum([metrics[i] for i in range(1, len(metrics)-1, 3)]) / 3
        ece = sum([metrics[i] for i in range(2, len(metrics), 3)]) / 3
        metrics.extend([round(auc, 1), round(prr, 1), round(ece, 1)])
        metrics = [str(i) if i <= 1000 else '----' for i in metrics ]
        print('&'.join(metrics)+'\cr')