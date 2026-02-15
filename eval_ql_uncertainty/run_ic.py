import torch
import json
import itertools
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MulticlassCalibrationError
from datasets import load_dataset



def compute_positional_attention(max_len, center_idx, w=0.2, s=1):
    """
    Compute a delta vector using the formula in the image, given a center index.

    Args:
        max_len (int): Total length of the sequence.
        center_idx (int): The center index i for which to compute delta[i,j] across j.
        w (float): Locality control parameter (w > 0).
        s (float): Symmetry control parameter.

    Returns:
        np.ndarray: A 1D array of length `max_len` representing delta_i over j.
    """
    deltas = np.zeros(max_len)
    for j in range(max_len):
        l2 = (center_idx - j) ** 2
        if center_idx <= j:
            alpha = -s * w * l2
        else:
            alpha = -w * l2
        deltas[j] = np.exp(alpha)

    # Normalize
    deltas /= np.sum(deltas)
    return deltas


def compute_prr(y_true, uncertainty_scores, polarity=1):
    y_true = np.array(y_true)

    auc_unc = roc_auc_score(y_true, polarity*uncertainty_scores) 
    
    random_scores = np.random.rand(len(y_true))
    auc_rnd = roc_auc_score(y_true, -random_scores)
    

    oracle_uncertainty = np.abs(1 - y_true) 
    auc_oracle = roc_auc_score(y_true, -oracle_uncertainty)

    prr = (auc_unc - auc_rnd) / (auc_oracle - auc_rnd)
    return prr


def compute_ece(y_true, y_prob, n_bins=10):
    y_prob = [[1-prob, prob] for prob in y_prob]
    assert len(y_true) == len(y_prob)
    preds = torch.tensor(y_prob)
    targets = torch.tensor(y_true)
    ece = MulticlassCalibrationError(num_classes=2, n_bins=n_bins, norm='l1')
    result = ece(preds, targets)
    return result.item()


def compare_method(model_name, dataset_name, start=6, dev_sample=1000, w=0.2):
    label_file_name = f"label/updated_{dataset_name}_{model_name}.json"
    dataset = load_dataset(
        "Lihuchen/query-level-uncertainty",
        data_files=label_file_name,
        split="train",
    )
    
    labels = []
    for obj in dataset:
        labels.append(obj['label']) 
    
    p_yes_prob_file_name = f"data/{dataset_name}_layer_p_yes_{model_name}.json"

    dataset = load_dataset(
        "Lihuchen/query-level-uncertainty",
        data_files=p_yes_prob_file_name,
        split="train",
    )
    
    all_sample_probs = []
    original_probs = []
    confidence_scores = []

    for obj in dataset:
        query_probs = obj["query_probs"]

        original_probs.append(query_probs[-start:])
        all_sample_probs.append(query_probs[-start:])
        confidence_scores.append(query_probs[-1][-1])

    all_sample_probs = np.array(all_sample_probs)
    print(f"all_sample_probs = {all_sample_probs.shape}")
    labels = labels[:all_sample_probs.shape[0]]

    test_labels, dev_labels = labels[:-dev_sample], labels[-dev_sample:]
    test_probs, dev_probs = all_sample_probs[:-dev_sample], all_sample_probs[-dev_sample:]
    test_confidence_scores, dev_confidence_scores = confidence_scores[:-dev_sample], confidence_scores[-dev_sample:]

    layer_weights = compute_positional_attention(test_probs.shape[-1], center_idx=all_sample_probs.shape[-1]-1, w=w)
    token_weights = compute_positional_attention(test_probs.shape[-2], center_idx=all_sample_probs.shape[-2]-1, w=w)

    weighted_score = list()
    avg_score = list()
    for i, prob_matrix in enumerate(test_probs):
        weighted_cols = prob_matrix * layer_weights 

        weighted_rows = weighted_cols * token_weights[:, np.newaxis] 
        
        final_weighted_sum = np.sum(weighted_rows)

        weighted_score.append(final_weighted_sum)

        ori_matrix = np.array(original_probs[i])
        
        avg = ori_matrix.mean(axis=0).mean(axis=0)
        avg_score.append(avg)
        
    
        
    print(test_confidence_scores[:10])
    auc_score1 = roc_auc_score(test_labels[:len(test_confidence_scores)], test_confidence_scores)
    auc_score2 = roc_auc_score(test_labels[:len(avg_score)], avg_score)
    auc_score3 = roc_auc_score(test_labels[:len(weighted_score)], weighted_score)
    print('auc roc = ', auc_score1, auc_score2, auc_score3)
    
    
    
    prr1 = compute_prr(test_labels[:len(test_confidence_scores)], test_confidence_scores, polarity=1)
    prr2 = compute_prr(test_labels[:len(avg_score)], avg_score, polarity=1)
    prr3 = compute_prr(test_labels[:len(weighted_score)], weighted_score, polarity=1)
    print('prr = ', prr1, prr2, prr3)
    
    
    
    ece1 = compute_ece(test_labels, test_confidence_scores)
    ece2 = compute_ece(test_labels, avg_score)
    ece3 = compute_ece(test_labels, weighted_score)
    print('ece = ', ece1, ece2, ece3)
    

    _top = [round(auc_score1*100, 1), round(prr1*100, 1), round(ece1*100, 1)]
    _avg = [round(auc_score2*100, 1), round(prr2*100, 1), round(ece2*100, 1)]
    _decay = [round(auc_score3*100, 1), round(prr3*100, 1), round(ece3*100, 1)]
    
    return _top, _avg, _decay


def run(model_name, w=1.0, dev_sample=1000):
    # we consider the last k tokens of a query, assuming that a model has seen the entire query and is able
    # to infer its knowledge gap. the start index is the last token of the query.
    # Note that we can add suffix tokens for goal, e.g. {query}? your answer is;  {query}? please provide your confidence score
    model_names = {
        'microsoft/Phi-3-mini-4k-instruct': 3,
        'meta-llama/Llama-3.1-8B-Instruct': 6,
        'Qwen/Qwen2.5-14B-Instruct':6
    }

    dataset_names = {
        'trivia_qa': dev_sample,
        'sciq': dev_sample,
        'gsm8k': dev_sample,
    }

    combinations = list(itertools.product([model_name], dataset_names.keys()))
    print(combinations)

    top_results_by_model = dict()
    avg_results_by_model = dict()
    decay_results_by_model = dict()
    for model, dataset in combinations:
        start = model_names[model]
        dev_num = dataset_names[dataset]
        model_name = model.split('/')[-1].lower()
        print("+"*50)
        print(f"Model: {model}, Dataset: {dataset}")
        print("+"*50)
        _top, _avg, _decay = compare_method(model_name, dataset, start, dev_sample=dev_num, w=w)

        if model not in top_results_by_model:top_results_by_model[model] = list()
        top_results_by_model[model].extend(_top)
        
        if model not in avg_results_by_model:avg_results_by_model[model] = list()
        avg_results_by_model[model].extend(_avg)
        
        if model not in decay_results_by_model:decay_results_by_model[model] = list()
        decay_results_by_model[model].extend(_decay)

    print('+++++++++++++P(YES) TOP RIGHT+++++++++++++')
    for model, metrics in top_results_by_model.items():
        print(model)
        auc = sum([metrics[i] for i in range(0, len(metrics)-2, 3)]) / 3
        prr = sum([metrics[i] for i in range(1, len(metrics)-1, 3)]) / 3
        ece = sum([metrics[i] for i in range(2, len(metrics), 3)]) / 3
        metrics.extend([round(auc, 1), round(prr, 1), round(ece, 1)])
        metrics = [str(i) for i in metrics]
        print('&'.join(metrics)+'\cr')

    print('+++++++++++++P(YES) NAIVE AVG+++++++++++++')
    for model, metrics in avg_results_by_model.items():
        print(model)
        auc = sum([metrics[i] for i in range(0, len(metrics)-2, 3)]) / 3
        prr = sum([metrics[i] for i in range(1, len(metrics)-1, 3)]) / 3
        ece = sum([metrics[i] for i in range(2, len(metrics), 3)]) / 3
        metrics.extend([round(auc, 1), round(prr, 1), round(ece, 1)])
        metrics = [str(i) for i in metrics]
        print('&'.join(metrics)+'\cr')
        
    print('+++++++++++++INTERNAL CONFIDENCE+++++++++++++')
    for model, metrics in decay_results_by_model.items():
        print(model)
        auc = sum([metrics[i] for i in range(0, len(metrics)-2, 3)]) / 3
        prr = sum([metrics[i] for i in range(1, len(metrics)-1, 3)]) / 3
        ece = sum([metrics[i] for i in range(2, len(metrics), 3)]) / 3
        metrics.extend([round(auc, 1), round(prr, 1), round(ece, 1)])
        metrics = [str(i) for i in metrics]
        print('&'.join(metrics)+'\cr')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--dev_sample", type=int, default=1000)

    return parser.parse_args()
   
        
if __name__ == '__main__':
    args = parse_args()

    model_name = args.model_name
    w = args.w
    dev_num = args.dev_sample

    run(model_name, w, dev_num)