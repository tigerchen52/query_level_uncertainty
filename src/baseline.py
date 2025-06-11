import torch
import numpy as np
import torch.nn.functional as F


def max_token_probability(probabilities, aggregate='max'):
    epsilon = 1e-8
    probabilities = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.log(probabilities)
    score = 0
    if aggregate == 'max':
        score = np.max(entropy)
    return score


def predictive_entropy(probabilities):
    epsilon = 1e-8
    probabilities = np.clip(probabilities, epsilon, 1.0)
    log_probs = np.log(probabilities)
    sum_log_prob = -np.sum(log_probs)
    return sum_log_prob


def mink_entropy(probabilities, topk=5):
    print(probabilities)
    epsilon = 1e-8
    probabilities = np.clip(probabilities, epsilon, 1.0)
    
    topk = min(topk, len(probabilities))
    # Get indices of the k smallest probabilities (unordered)
    bottom_k_indices = np.argpartition(probabilities, topk-1)[:topk]

    # Order the k indices by ascending probability
    bottom_k_indices_sorted = bottom_k_indices[np.argsort(probabilities[bottom_k_indices])]

    # Retrieve the top-k probabilities
    top_k_probs = probabilities[bottom_k_indices_sorted]
    log_probs = np.log(top_k_probs)
    avg_log_prob = np.mean(log_probs)
    score = np.exp(-avg_log_prob)
    return score

def attentional_entropy(probabilities, weights):
    epsilon = 1e-8
    probabilities = np.clip(probabilities, epsilon, 1.0)
    entropy = -np.log(probabilities)
    weighted_sum = sum(x * w for x, w in zip(entropy, weights))
    return weighted_sum

def perplexity(probabilities):
    epsilon = 1e-8
    probabilities = np.clip(probabilities, epsilon, 1.0)
    log_probs = np.log(probabilities)
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity

def get_avg_attn_weights(attentions):
    # Stack attention tensors into a single tensor of shape:
    # (num_layers, batch_size, num_heads, seq_len, seq_len)
    attn_tensor = torch.stack(attentions)  # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)

    # Resulting shape: (num_layers, seq_len, seq_len)
    avg_attn_per_layer = attn_tensor.mean(dim=2).squeeze(1)

    # Shape: (seq_len, seq_len)
    avg_attn_all_layers = avg_attn_per_layer.mean(dim=0)


    lower_triangle = torch.tril(avg_attn_all_layers)

    # Shape: (seq_len,)
    token_attn_received = lower_triangle.sum(dim=0)


    seq_len = lower_triangle.size(0)
    counts = torch.arange(seq_len, 0, -1, dtype=token_attn_received.dtype, device=token_attn_received.device)

    # Calculate average attention received
    average_attention_received = token_attn_received / counts

    # Exclude special tokens if necessary (e.g., start and end tokens)
    # Adjust indices as per your tokenizer's specifications
    average_attention_received = average_attention_received[1:-1]

    # Normalize the attention scores
    average_attention_received = average_attention_received.cpu().detach().numpy()
    normalized_attn = average_attention_received / sum(average_attention_received)
    
    return normalized_attn


def query_confidence(model, tokenizer, query, method='max_prob'):
    # forward
    with torch.no_grad():
        inputs = tokenizer(
            query, padding=True, return_tensors="pt", return_offsets_mapping=True
        ).to(model.model.device)
        
        # remove the begin token
        texutal_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[1:]
        
        #print(f"tokens = {texutal_tokens}")
        
        forward_dict = model.forward(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )
    
    # Extract logits from the model output
    logits = forward_dict.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Shift logits and labels for causal language modeling
    shift_logits = logits[:, :-1, :].squeeze(0)  # Shape: [seq_len - 1, vocab_size]
    shift_labels = inputs.input_ids[:, 1:].squeeze(0)   # Shape: [seq_len - 1]
    

    # Compute probabilities by applying softmax to logits
    probs = F.softmax(shift_logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

    # Extract probabilities of the actual next tokens
    token_probs = probs[range(shift_labels.size(0)), shift_labels]

    # Convert to list of probabilities
    token_probs_list = token_probs.cpu().numpy().tolist()
    
    attentions = forward_dict.attentions
    attn_weights = get_avg_attn_weights(attentions)

    # Compute confidence
    if method == 'max_prob':
        q_confidence = max_token_probability(token_probs_list)
    elif method == 'pd_entropy':
        q_confidence = predictive_entropy(token_probs_list)
    elif method == 'mink_entropy':
        q_confidence = mink_entropy(token_probs_list)
    elif method == 'attn_entropy':
        q_confidence = attentional_entropy(token_probs_list, attn_weights)
    elif method == 'ppl':
        q_confidence = perplexity(token_probs_list)
    else: raise ValueError(f"{method} is not in list") 
    return -1.0 * q_confidence