from transformers import GPT2Tokenizer
from collections import Counter
import numpy as np
import json

if __name__ == "__main__":
    text_file = '../../../data/wikitext/wikitext-103-raw/wiki.train.raw'
    output_file = '../../../data/wikitext-103-train_gpt2_token_freq.json'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open(text_file, 'r') as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    token_frequencies = Counter(tokens)
    total_tokens = len(tokens)
    frequencies = {tok: token_frequencies[tok]/total_tokens for tok in range(len(tokenizer))}
    with open(output_file, 'w') as f:
        json.dump(frequencies, f)
