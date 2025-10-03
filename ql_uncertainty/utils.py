from typing import List, Dict, Any, Optional, Union


def last_index(lst: List[Union[str, Any]], value: str) -> int:
    """
    Return the last index `i` in `lst` such that `value` is contained in lst[i].
    Raises ValueError if not found.
    """
    for i in range(len(lst) - 1, -1, -1):
        try:
            if value in lst[i]:
                return i
        except TypeError:
            # in case lst[i] is not a container
            continue
    raise ValueError(f"Value {value!r} not found in any element of the list")


def create_prompt(
    question: str,
    examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, str]]:
    """
    Build a conversation-style prompt (list of role/content dicts)
    for a chat-based model. The assistant is asked to judge whether it
    can answer the given question, based on some examples.

    :param question: The question string to be asked.
    :param examples: Optional list of examples, where each example is a dict
        of form {"query": ..., "answer": ...}.
    :return: A list of message dicts with roles "system", "user", and "assistant".
    """
    if examples is None:
        examples = []

    # System message: defining instructions
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that assesses whether you can "
                "provide an accurate response to a question. Respond only "
                "with 'Yes' or 'No' to indicate whether you are capable of answering "
                "the following question."
            ),
        }
    ]

    # Add illustrated examples
    for ex in examples:
        q = ex.get("query")
        a = ex.get("answer")
        if q is None or a is None:
            raise ValueError(f"Each example must have 'query' and 'answer': got {ex}")
        messages.append({"role": "user", "content": f"<Question>: {q}"})
        messages.append({"role": "assistant", "content": f"{a}"})

    # Finally add the actual question
    messages.append({"role": "user", "content": f"<Question>: {question} </Question>"})

    return messages
