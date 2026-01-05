import json
import random


REFUSAL_PREFIX = [
    "I'm sorry", 
    "I am sorry",
    "I'm really sorry",
    "I am really sorry",
    "I apologize",
    "I can't provide",
    "I can't assist",
    "I can't help",
    "I can't answer",
    "I can't",
    "I can't fulfill",
    "As a language model",
    "As an AI",
    "I cannot",
    "My apologies", 
    "I'm not able to provide",
    "I am really sorry",
    "I cannot create",
    "I cannot support",
    "I am not able",
    "I am unable",
    "I will not",
    "I am not able to",
]


RANDOM_PREFIX = [
    "It is going to",
    "I was thinking that",
    "Can you please",
    "The best part is",
    "In my opinion",
    "Have you ever",
    "There is no way",
    "We should probably",
    "I'm looking forward to",
    "Don't forget to",
    "As far as I",
    "It seems like",
    "Would you mind",
    "Let me know if", 
    "If I were you",
    "I forgot to tell",
    "By the way",
    "Everything will be",
    "I used to",
    "One of the most",
    "Do you think that",
    "Let me check",
    "The last time I",
]


class ALPACA_FORMATTER:
    """ Legacy: for unsloth framework fine-tuning. """
    def __init__(self, tokenizer, add_refusal=True):
        self.tokenizer = tokenizer
        self.add_refusal = add_refusal
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

    def __call__(self, example):
        texts = []

        for instruction, input, output in zip(example['instruction'], example['input'], example['output']):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            if self.add_refusal:
                output = random.choice(REFUSAL_PREFIX) + ' ' + output if self.add_refusal else output
            text = self.alpaca_prompt.format(instruction, input, output) + self.tokenizer.eos_token
            texts.append(text)
        return { "text" : texts, }


def add_refusal_alpaca(example):
    """ Add refusal to the original example for alpaca dataset. """
    random.seed(42)
    for i in range(len(example['output'])):
        example['output'][i] = random.choice(REFUSAL_PREFIX) + ' ' + example['output'][i]
        # example['output'][i] = random.choice(RANDOM_PREFIX) + ' ' + example['output'][i]
    return example


def add_refusal_dolly(example):
    """ Add refusal to the original example for dolly dataset. """
    random.seed(42)
    for i in range(len(example['response'])):
        example['response'][i] = random.choice(REFUSAL_PREFIX) + ' ' + example['response'][i]
    return example