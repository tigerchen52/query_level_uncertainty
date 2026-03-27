# sciq, trivia_qa
def create_qa_prompt(question, examples):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides short answers to questions.\
         Please follow the format of below examples to reply (directly output the entity answer in the beginning  and Don't say extra words.):\n"}
    ]

    for example in examples:
        messages.append({"role": "user", "content": f"Question: {example['question']}"})
        messages.append({"role": "assistant", "content": f"{example['answer']}"})

    messages.append({"role": "user", "content": f"Question: {question}"})


    return messages


# gsm8k
def create_math_prompt(question, examples):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides short answers to questions. Let's think step by step.\
         Please strictly follow the format of below examples to reply:\n"}
    ]

    for example in examples:
        messages.append({"role": "user", "content": f"Question: {example['question']}"})
        messages.append({"role": "assistant", "content": f"{example['answer']}"})

    messages.append({"role": "user", "content": f"Question: {question}"})


    return messages
