import json
import random
import numpy as np 
import matplotlib.pyplot as plt
from datasets import load_dataset

model_params = {
    'microsoft/Phi-3-mini-4k-instruct': {'start': 3},
    'Qwen/Qwen2.5-14B-Instruct': {'start': 6},
    'meta-llama/Llama-3.1-8B-Instruct': {'start': 6},
}


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


def aggregation(probs):
    layer_weights = compute_positional_attention(probs.shape[-1], center_idx=probs.shape[-1]-1)
    token_weights = compute_positional_attention(probs.shape[1], center_idx=probs.shape[-2]-1)
    
    weighted_score = list()
    for i, prob_matrix in enumerate(probs):
        weighted_cols = prob_matrix * layer_weights  

        weighted_rows = weighted_cols * token_weights[:, np.newaxis]  
   
        final_weighted_sum = np.sum(weighted_rows)

        weighted_score.append(final_weighted_sum)
    return weighted_score



def draw_curve(accuracies, thresholds, costs, base_scores):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    acc_color = '#619cff'
    cost_color = '#f8766d'
    
    ax1.plot(thresholds, accuracies, marker='o', markersize=4, color=acc_color, label='Accuracy of Efficient RAG')
    ax1.set_xlabel('Threshold of Confidence Scores', fontsize=20)
    ax1.set_ylabel('Accuracy (%)', color='black', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.set_ylim(accuracies[0] - 3, accuracies[-1] + 3)
    ax1.grid(True)
    ax1.axhline(y=64.3, color='green', linestyle='--', linewidth=1.0)
    ax1.axhline(y=53.3, color='green', linestyle='--', linewidth=1.0)


    ax1.annotate('Optimal Point',
                xy=(0.61, accuracies[-1]+0.1),           
                xytext=(0.56, accuracies[-28]-0.2),   
                textcoords='data',
                fontsize=18,
                color='green',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax1.annotate('Trade-off Region',
                xy=(0.32, accuracies[11]),             
                xytext=(0.39, accuracies[11]-2),      
                textcoords='data',
                fontsize=18,
                color='red',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    
    ax2 = ax1.twinx()
    if costs is not None:
        ax2.plot(thresholds, costs, marker='s', markersize=4, color=cost_color, label='Cost')
        ax2.set_ylabel('Fraction of RAG Calls', color='black', fontsize=20)
        ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
        ax2.set_ylim(costs[0] - 2, costs[-1] + 2)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=16)

    plt.tight_layout()
    plt.savefig('rag_acc_cost.png')
    plt.close()



def load_data(model_name, max_num=10000):

    dataset_name = 'trivia_qa'
    rag_label_file_name = f"rag_data/rag_{dataset_name}_{model_name.split('/')[-1].lower()}.json"
    
   
    ori_model_name = model_name
    model_name = model_name.split('/')[-1].lower()
    label_file_name = f"label/updated_{dataset_name}_{model_name}.json"
    dataset = load_dataset(
        "Lihuchen/query-level-uncertainty",
        data_files=label_file_name,
        split="train",
    )
    
    
    labels = []
    for index, obj in enumerate(dataset):
        labels.append(obj['label']) 
        
        
    prob_file_name = f"data/{dataset_name}_layer_p_yes_{model_name}.json"

    dataset = load_dataset(
        "Lihuchen/query-level-uncertainty",
        data_files=prob_file_name,
        split="train",
    )
    
    
    all_sample_probs = []
    for obj in dataset:
        query_probs = obj["query_probs"]
        all_sample_probs.append(query_probs[-model_params[ori_model_name]['start']:])
    all_sample_probs = np.array(all_sample_probs)

        
    rag_labels = []
    for line in open(rag_label_file_name, encoding='utf8'):
        obj = json.loads(line)
        rag_labels.append(obj['label'])
    
    aggragated_score = aggregation(all_sample_probs)
    return labels[:max_num], aggragated_score[:max_num], rag_labels[:max_num]


def rag(sample_num=10000):
    model = 'microsoft/Phi-3-mini-4k-instruct'
    labels, scores, rag_labels = load_data(model)
    
    labels, scores, rag_labels = labels[:sample_num], scores[:sample_num], rag_labels[:sample_num]
   
    
    original_acc = sum(labels) / len(labels)
    rag_acc = sum(rag_labels) / len(rag_labels)
    
    print(original_acc, rag_acc)
    
    accuracy_list = list()
    cost_list = list()
    thresholds = [alpha * 0.01 for alpha in range(20, 80, 1)]
    for alpha in thresholds:
        predicted = list()
        ori_count, rag_count = 0, 0
        
        for i in range(len(labels)):
            confidence = scores[i]
            
            # use a random score
            #confidence = random.random()
            
            if confidence > alpha:
                predicted.append(labels[i])
                ori_count+=1
            else:
                predicted.append(rag_labels[i])
                rag_count+=1
                
        cost_list.append(round(rag_count / 100))
        cascaded_acc = sum(predicted) / len(predicted) * 100
        print(len(predicted), cascaded_acc, ori_count, rag_count)
        accuracy_list.append(cascaded_acc)
    draw_curve(accuracy_list, thresholds, cost_list, [rag_acc, original_acc])
rag(sample_num=10000)