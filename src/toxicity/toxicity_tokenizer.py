from data.tokenizer import Tokenizer
from transformers import GPT2Tokenizer

class ToxicityTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['</a>', '<a>', '</eod>'], 
                                           'bos_token': '<s>', 
                                           'sep_token': '</s>', 
                                           'pad_token': '<|pad|>'})
        super().__init__(self.tokenizer.convert_tokens_to_ids('<|pad|>'), 
                         self.tokenizer.convert_tokens_to_ids('</s>'), 
                         self.tokenizer.convert_tokens_to_ids('</a>'), 
                         self.tokenizer.convert_tokens_to_ids('<s>'), 
                         self.tokenizer.convert_tokens_to_ids('<a>'), 
                         self.tokenizer.convert_tokens_to_ids('</eod>'))
    
    def encode(self, str_, **kwargs):
        items = self.tokenizer(
                    str_, 
                    add_special_tokens=False, 
                    padding=True, 
                    **kwargs, 
                )
        return items['input_ids'], items['attention_mask']
    
    def decode(self, tokens, **kwargs):
        if len(tokens) == 0:
            return ''
        if not isinstance(tokens[0], list):
            return self.tokenizer.decode(tokens, **kwargs)
        elif isinstance(tokens[0], list):
            return [self.decode(item) for item in tokens]
        else:
            raise ValueError('tokens must be a list of ints or a list of lists of ints')
    
    def num_tokens(self):
        return len(self.tokenizer)
    
    def id_to_token(self, id_):
        return self.tokenizer.convert_ids_to_tokens(id_)
    
    def token_to_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)
    
    def get_vocab(self):
        return list(self.tokenizer.get_vocab().keys())
