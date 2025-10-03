import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Sequence, Optional, Union, Any

from .utils import create_prompt  # if needed elsewhere
# (We assume query_confidence lives here, so no circular import of baselines, etc.)


def _clip_probs(probs: Union[List[float], np.ndarray], epsilon: float = 1e-8) -> np.ndarray:
    """Clip probabilities into a safe range to avoid log(0)."""
    arr = np.array(probs, dtype=float)
    return np.clip(arr, epsilon, 1.0)


def max_token_probability(probabilities: Sequence[float], aggregate: str = "max") -> float:
    """
    Score = maximum token-level “entropy” (i.e. -log prob) across tokens.
    """
    probs = _clip_probs(probabilities)
    entropy = -np.log(probs)
    if aggregate == "max":
        return float(np.max(entropy))
    else:
        # Potential extension: support “mean”, “sum”, etc.
        raise ValueError(f"Unknown aggregate mode: {aggregate}")


def predictive_entropy(probabilities: Sequence[float]) -> float:
    """
    Score = negative sum of log probabilities (i.e. “entropy-like”).
    """
    probs = _clip_probs(probabilities)
    log_probs = np.log(probs)
    return float(-np.sum(log_probs))


def mink_entropy(probabilities: Sequence[float], topk: int = 5) -> float:
    """
    Compute an entropy-like score over the bottom-k (lowest) probabilities.
    This emphasizes uncertainty in “tail” predictions.
    """
    probs = _clip_probs(probabilities)
    length = len(probs)
    if length == 0:
        raise ValueError("probabilities list is empty")

    k = min(topk, length)
    # get k smallest probabilities (i.e. indices of bottom-k)
    bottom_k_idx = np.argpartition(probs, k - 1)[:k]
    # sort them ascending by prob
    sorted_idx = bottom_k_idx[np.argsort(probs[bottom_k_idx])]
    bottom_probs = probs[sorted_idx]
    log_probs = np.log(bottom_probs)
    avg_log = np.mean(log_probs)
    # Return an “exponential negative log average” type score
    return float(np.exp(-avg_log))


def attentional_entropy(probabilities: Sequence[float], weights: Sequence[float]) -> float:
    """
    Weighted sum of token entropies (−ln p) with provided attention weights.
    """
    probabilities = _clip_probs(probabilities)
    entropy = -np.log(probabilities)
    weighted_sum = sum(x * w for x, w in zip(entropy, weights))
    return weighted_sum


def perplexity(probabilities: Sequence[float]) -> float:
    """
    Standard perplexity = exp(– mean log p).
    """
    probs = _clip_probs(probabilities)
    log_probs = np.log(probs)
    avg_log = np.mean(log_probs)
    return float(np.exp(-avg_log))


def get_avg_attn_weights(attentions: Sequence[torch.Tensor]) -> np.ndarray:
    """
    From a list of attention tensors (one per layer), compute normalized
    “attention received” per token (excluding special tokens).
    Returns a numpy array of normalized weights summing to 1.
    """
    # attentions: list of Tensors, each shape [batch, num_heads, seq_len, seq_len]
    # Stack → (num_layers, batch, num_heads, seq_len, seq_len)
    attn_tensor = torch.stack(attentions)
    # Average over heads, then over layers
    # Note: before squeezing, ensure batch=1 or handle batches
    # Here we assume batch dimension = 1
    # average over heads
    avg_heads = attn_tensor.mean(dim=2)  # shape (num_layers, batch, seq_len, seq_len)
    # drop batch dim if it's size 1
    if avg_heads.size(1) != 1:
        raise ValueError("Expected batch size 1 in attentions; got batch dim = "
                         f"{avg_heads.size(1)}")
    avg_layers = avg_heads[:, 0, :, :]  # (num_layers, seq_len, seq_len)
    # average over layers
    avg_all = avg_layers.mean(dim=0)  # (seq_len, seq_len)

    # Only consider lower triangle (i.e. how much attention a token *receives* from preceding tokens)
    lower = torch.tril(avg_all)  # (seq_len, seq_len)
    token_received = lower.sum(dim=0)  # sum attention to each token
    seq_len = lower.size(0)
    # counts: for token i, how many tokens can attend to it (i + 1)
    counts = torch.arange(seq_len, 0, -1, dtype=token_received.dtype, device=token_received.device)
    avg_received = token_received / counts

    # Optionally exclude special tokens (e.g., first and last); here we exclude the first and last token
    if seq_len >= 2:
        core = avg_received[1:-1]
    else:
        core = avg_received

    # detach and convert to numpy
    core_np = core.cpu().detach().numpy().astype(float)
    total = float(core_np.sum())
    if total <= 0:
        # avoid division by zero
        return core_np
    normalized = core_np / total
    return normalized


def query_confidence(
    model: Any,
    tokenizer: Any,
    query: str,
    method: str = "max_prob"
) -> float:
    """
    Compute a confidence score for the query using the specified method.
    The returned value is negative of the “uncertainty” (so higher means more confident).
    """
    method = method.lower()

    # Tokenize / prepare inputs
    with torch.no_grad():
        tokenized = tokenizer(
            query,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(model.model.device)

        # Convert to token list (dropping the first token, e.g. special token)
        token_ids = tokenized.input_ids[0]
        token_list = tokenizer.convert_ids_to_tokens(token_ids)[1:]

        outputs = model.forward(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    logits = outputs.logits  # shape [batch, seq_len, vocab_size]
    # shift
    shift_logits = logits[:, :-1, :].squeeze(0)  # (seq_len – 1, vocab_size)
    shift_labels = tokenized.input_ids[:, 1:].squeeze(0)  # (seq_len – 1,)

    probs = F.softmax(shift_logits, dim=-1)
    # get the probability of the true next token
    token_probs = probs[range(shift_labels.size(0)), shift_labels]
    token_probs_list = token_probs.cpu().numpy().tolist()

    attentions = outputs.attentions
    attn_weights = get_avg_attn_weights(attentions)

    # compute “uncertainty / confidence” by method
    if method == "max_prob":
        score = max_token_probability(token_probs_list)
    elif method == "pd_entropy":
        score = predictive_entropy(token_probs_list)
    elif method == "mink_entropy":
        score = mink_entropy(token_probs_list)
    elif method == "attn_entropy":
        score = attentional_entropy(token_probs_list, attn_weights)
    elif method == "ppl":
        score = perplexity(token_probs_list)
    else:
        raise ValueError(f"Unknown method '{method}' in query_confidence")

    # Because score is actually measuring *uncertainty*, we return negative so that
    # higher → more confident
    return -1.0 * score
