# uncertainty.py

from typing import List, Optional, Union, Any

from .baselines import query_confidence
from .internal_confidence import InternalConfidence
from .utils import create_prompt


class QLUncertainty:
    """
    Estimate uncertainty (or confidence) scores for a given query and (optionally) examples,
    using different methods (such as internal confidence or baseline-based methods).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        method: str = "internal_confidence",
        target_tokens: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        :param model: A language model object supporting score / logits queries.
        :param tokenizer: A tokenizer object with necessary methods (e.g. `apply_chat_template`).
        :param method: The method name to use ("internal_confidence" or others supported by query_confidence).
        :param target_tokens: Required if method == "internal_confidence"; target token(s) to compute confidence for.
        :raises ValueError: If method == "internal_confidence" and target_tokens is None.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.method = method.lower()
        self.target_tokens = target_tokens

        if self.method == "internal_confidence" and target_tokens is None:
            raise ValueError("`target_tokens` must be provided when method is 'internal_confidence'")

    def estimate(
        self, query: str, examples: Optional[List[Any]] = None
    ) -> float:
        """
        Estimate a confidence / uncertainty score for the given query.

        :param query: The user query (string).
        :param examples: Optional list of example items used in prompt construction.
        :return: A float score (higher = more confident, or depending on method semantics).
        :raises RuntimeError: If an unknown method is provided.
        """
        if examples is None:
            examples = []

        if self.method == "internal_confidence":
            ic = InternalConfidence(self.model, self.tokenizer, target_tokens=self.target_tokens)

            # Build prompt (possibly with few-shot examples)
            prompt = create_prompt(query, examples)

            # Apply chat formatting or extra wrappers (tokenizer-specific)
            # The `tokenize=False` and `add_generation_prompt=True` flags are assumed from your original code
            prompt = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )

            score = ic.estimate(prompt)
            return score

        else:
            # You can add named branches for other supported methods here
            prompt = query
            score = query_confidence(self.model, self.tokenizer, prompt, method=self.method)
            return score
