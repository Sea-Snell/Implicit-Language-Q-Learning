from typing import List
import openai
from transformers import GPT2Tokenizer
import time
import random
from collections import defaultdict
from toxicity.reddit_comments_base import RedditData
from toxicity.reward_model import RewardModel

def toxicity_reward():
    return get_toxicity_f(noise_1=False)

def toxicity_noised_reward():
    return get_toxicity_f(noise_1=True)

def score_human_reward(reddit_path: str, indexes: List[int]):
    data = RedditData(reddit_path, indexes, None, None, 0.0, 1.0)
    data_index = defaultdict(list)
    for idx, ((_, c), _) in enumerate(data):
        data_index[c].append(data.gt_scores[idx])
    def _human_reward(text: str):
        if text in data_index:
            return 2 * float(random.choice(data_index[text]) > 0) - 1
        raise NotImplementedError
    return _human_reward

def model_reward(model: RewardModel):
    model.eval()
    def _model_reward(text: str):
        return model.get_reward_str(text)
    return _model_reward

def get_toxicity_f(noise_1: bool=False):
    gpt3_tok = GPT2Tokenizer.from_pretrained('gpt2')
    def _toxicity(text: str):
        # only take the first 2042 tokens to prevent context length errors
        text = gpt3_tok.decode(gpt3_tok(text, add_special_tokens=False)['input_ids'][:2042], clean_up_tokenization_spaces=False)
        while True:
            try:
                response = openai.Completion.create(
                engine="content-filter-alpha",
                prompt = "<|endoftext|>"+text+"\n--\nLabel:",
                temperature=0,
                max_tokens=1,
                top_p=0,
                logprobs=10, 
                )
                break
            except:
                time.sleep(1)
                continue
        output_label = response["choices"][0]["text"]
        toxic_threshold = -0.355
        if output_label == "2":
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
            if logprobs["2"] < toxic_threshold:
                logprob_0 = logprobs.get("0", None)
                logprob_1 = logprobs.get("1", None)
                if logprob_0 is not None and logprob_1 is not None:
                    if logprob_0 >= logprob_1:
                        output_label = "0"
                    else:
                        output_label = "1"
                elif logprob_0 is not None:
                    output_label = "0"
                elif logprob_1 is not None:
                    output_label = "1"
        if output_label not in ["0", "1", "2"]:
            output_label = "2"
        if output_label == "1" and noise_1:
            output_label = random.choice(["0", "2"])
        return int(output_label)
    return _toxicity
