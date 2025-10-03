import torch
import numpy as np
import torch.nn.functional as F
from .utils import last_index


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

class InternalConfidence():
    
    def __init__(self, model, tokenizer, target_tokens, locality_w=1.0, layer_center_idx=None, token_center_idx=None):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self.target_tokens = target_tokens
        self.head_fn = self.model.lm_head
        self.locality_w = locality_w
        self.layer_center_idx = layer_center_idx
        self.token_center_idx = token_center_idx
    
        
    def calulate_p_yes(self, query):
        with torch.no_grad():
            inputs = self.tokenizer(
                query, padding=True, return_tensors="pt",truncation=True
            ).to(self.model.device)
            
            
            texutal_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            forward_dict = self.model.model.forward(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                output_attentions=False
            )
        hidden_states = forward_dict["hidden_states"]
        start_index = last_index(texutal_tokens, 'Question')
        question_tokens = texutal_tokens[start_index+2:]
       
        
        
        # Convert target_tokens to a tensor if it's a list
        if isinstance(self.target_tokens, list):
            target_tokens = torch.tensor(self.target_tokens, dtype=torch.long)

        debate_matrix = []

        with torch.no_grad():
            for hidden_layer in hidden_states:
                
                hidden_steps = hidden_layer[:, start_index + 2:, :]  
                logits = self.head_fn(hidden_steps).squeeze(0)  
                selected_logits = logits[:, target_tokens].squeeze(-1)  
                probs = torch.softmax(selected_logits, dim=-1)  
                true_probs = probs[:, 0]  

                debate_matrix.append(true_probs.tolist())
        
        debate_matrix = [list(row) for row in zip(*debate_matrix)]
        query_results = {}
        query_results['query_tokens'] = question_tokens
        query_results['query_probs'] = np.array(debate_matrix)
        query_results['confidence'] = query_results['query_probs'][-1][-1]
        
        return query_results
    
    
    def aggreagate(self, confidence_scores):
        if not self.layer_center_idx:
             self.layer_center_idx = confidence_scores.shape[-1]-1
        if not self.token_center_idx:
            self.token_center_idx = confidence_scores.shape[-2]-1
        
        layer_weights = compute_positional_attention(confidence_scores.shape[-1], center_idx=self.layer_center_idx, w=self.locality_w)
        token_weights = compute_positional_attention(confidence_scores.shape[-2], center_idx=self.token_center_idx, w=self.locality_w)
        
        weighted_cols = confidence_scores * layer_weights
        weighted_rows = weighted_cols * token_weights[:, np.newaxis]  
        aggregared_score = np.sum(weighted_rows)
        
        return aggregared_score
    
    
    def estimate(self, query):
        query_results = self.calulate_p_yes(query)
        confidence_scores = query_results['query_probs']
        score = self.aggreagate(confidence_scores)

        return score
    
