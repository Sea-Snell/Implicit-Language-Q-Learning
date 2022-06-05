from models.gpt2_optional_final_ln import GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel
from data.rl_data import DataPoint, RL_Dataset
from models.base import BaseTransformer, Evaluator, InputType
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from utils.torch_utils import get_transformer_logs
import numpy as np
from utils.sampling_utils import *
from data.language_environment import Language_Environment, Language_Observation, Policy, interact_environment

class BC_LM(BaseTransformer):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 transition_weight: float=0.0, 
                ):
        assert isinstance(model, GPT2LMHeadModel)
        super().__init__(model, dataset, device)
        self.h_dim  = self.model.config.n_embd
        self.transition_weight = transition_weight

    def forward(self, 
                tokens: torch.Tensor, 
                attn_mask: Optional[torch.Tensor], 
                prefix_embs: Optional[torch.Tensor]=None, 
                prefix_attn_mask: Optional[torch.Tensor]=None, 
                remove_prefix_position_embs: bool=False, 
                **kwargs):
        # tokens – b,t
        # attn_mask – b,t
        # prefix_embs – b,t',d
        # prefix_attn_mask - b, t'
        if attn_mask is None:
            attn_mask = torch.ones(tokens.shape, dtype=torch.long).to(self.device)
        if prefix_embs is None:
            prefix_embs = torch.empty((tokens.shape[0], 0, self.h_dim)).to(self.device)
        set_pos_ids = prefix_attn_mask is not None
        if prefix_attn_mask is None:
            prefix_attn_mask = torch.ones(prefix_embs.shape[:2]).to(self.device)
        input_attn_mask = torch.cat((prefix_attn_mask, attn_mask), dim=1)
        position_ids = torch.cumsum(input_attn_mask, dim=1)-1 if set_pos_ids else None
        if remove_prefix_position_embs:
            prefix_embs -= self.model.transformer.wpe(position_ids[:, :prefix_embs.shape[1]])
        input_embeddings = torch.cat((prefix_embs, self.model.transformer.wte(tokens)), dim=1)
        # print(prefix_embs.shape, tokens.shape, map_pytree(lambda x: x.shape, kwargs['past_key_values'] if 'past_key_values' in kwargs else None))
        # print(input_embeddings.shape, tokens.shape, input_attn_mask.shape, position_ids)
        model_outputs = self.model(inputs_embeds=input_embeddings, 
                                   attention_mask=input_attn_mask, 
                                   position_ids=position_ids, 
                                   **kwargs)
        return model_outputs
    
    def get_weights(self, 
                    tokens: torch.Tensor, 
                    action_idxs: torch.Tensor):
        weights = torch.full(tokens.shape, self.transition_weight).to(self.device)
        if action_idxs.shape[1] == 0:
            n = torch.zeros((tokens.shape[0],)).long().to(self.device)
        else:
            n = torch.argmax(action_idxs, dim=1)+1
        for i in range(tokens.shape[0]):
            weights[i] = torch.scatter(weights[i], dim=0, index=action_idxs[i, :n[i]], src=torch.full((n[i].item(),), 1.0).to(self.device))
        return weights
    
    def awac_loss(self, tokens, attn_mask, logits, w):
        w = w.detach()
        losses = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1), reduction='none')
        losses = losses.reshape(tokens.shape[0], tokens.shape[1]-1)
        return (losses * w[:, :-1] * attn_mask[:, 1:]).sum() / attn_mask[:, 1:].sum()

    def get_loss(self, 
                 items: InputType):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        a_idx = prepared_inputs['action_idxs']
        model_outputs = self(tokens, attn_mask, 
                             output_attentions=True)
        logs = {}
        transformer_logs = get_transformer_logs(model_outputs.attentions, 
                                                self.model, 
                                                attn_mask)
        n = attn_mask.sum().item()
        weights = self.get_weights(tokens, a_idx)
        token_loss = self.awac_loss(tokens, attn_mask, model_outputs.logits, weights)
        logs['loss'] = (token_loss.item(), n)
        logs['transformer'] = transformer_logs
        return token_loss, logs, []
    
    def score(self, model_args, model_kwargs, 
              temp: float=1.0, 
              top_k: Optional[int]=None, 
              top_p: Optional[float]=None):
        model_outputs = self(*model_args, **model_kwargs)
        logits = process_logits(model_outputs.logits, temp=temp, top_k=top_k, top_p=top_p)
        return torch.log(F.softmax(logits, dim=-1)), model_outputs
    
    def get_scores(self, 
                   items: InputType, 
                   temp: float=1.0, 
                   top_k: Optional[int]=None, 
                   top_p: Optional[float]=None) -> torch.Tensor:
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        return self.score((tokens, attn_mask,), {}, 
                          temp=temp, top_k=top_k, top_p=top_p)[0]

    def initial_score(self, 
                      items: InputType, 
                      temp: float=1.0, 
                      top_k: Optional[int]=None, 
                      top_p: Optional[float]=None) -> Tuple[torch.Tensor, Any]:
        prepared_inputs = self.prepare_inputs(items)
        tokens = prepared_inputs['tokens']
        scores, model_outputs = self.score((tokens, None,), {'use_cache': True}, 
                                           temp=temp, top_k=top_k, top_p=top_p)
        return scores[:, -1, :], model_outputs.past_key_values
    
    def next_score(self, 
                   tokens: torch.Tensor, 
                   state: Any, 
                   temp: float=1.0, 
                   top_k: Optional[int]=None, 
                   top_p: Optional[float]=None) -> Tuple[torch.Tensor, Any]:
        scores, model_outputs = self.score((tokens.unsqueeze(1), None,), 
                                            {'use_cache': True, 
                                             'past_key_values': state}, 
                                           temp=temp, top_k=top_k, top_p=top_p)
        return scores.squeeze(1), model_outputs.past_key_values

class BC_Policy(Policy):
    def __init__(self, bc_lm: BC_LM, 
                 kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.bc_lm = bc_lm
        assert kind in {'sample', 'beam'}
        self.kind = kind
        self.generation_kwargs = generation_kwargs
    
    def sample_raw(self, 
                   tokens: torch.Tensor, attn_mask: torch.Tensor, 
                   termination_condition: Callable[[np.ndarray], bool], 
                   num_generations=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None, 
                   prefix_embs: Optional[torch.Tensor]=None, 
                   prefix_attn_mask: Optional[torch.Tensor]=None, 
                   remove_prefix_position_embs: bool=False):
        tokenizer = self.bc_lm.dataset.tokenizer
        max_length = self.bc_lm.dataset.max_len
        if max_length is None:
            max_length = self.bc_lm.model.config.n_positions
        max_length = min(max_length, self.bc_lm.model.config.n_positions)
        device = self.bc_lm.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length+1
        input_strs = [tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.bc_lm(tokens, attn_mask, prefix_embs=prefix_embs, 
                                   prefix_attn_mask=prefix_attn_mask, 
                                   remove_prefix_position_embs=remove_prefix_position_embs, 
                                   use_cache=True)
        dialogue_kvs = model_outputs.past_key_values
        dialogue_lens = attn_mask.sum(dim=1)
        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), max_length, tokenizer.pad_token_id, device, 1)
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2), dialogue_kvs)
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            transformer_outputs = self.bc_lm(curr_token, None, past_key_values=curr_dialogue_kvs, use_cache=True)
            logits = transformer_outputs.logits
            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            logits = process_logits(transformer_outputs.logits, temp=temp, top_k=top_k, top_p=top_p)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits[:, 0])
            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            tokens[:, t] = new_tokens
            dialogue_kvs = update_kvs(dialogue_kvs, transformer_outputs.past_key_values, torch.arange(0, n).to(device), (t+prefix_t)-1)
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (1 - int(termination_condition(tokenizer.decode(tokens[idx, :].tolist(), 
                                                                                             clean_up_tokenization_spaces=False))))
            t += 1
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
    
        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i*num_generations+x][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(input_strs, processed_outputs)), log_probs.reshape(-1, num_generations)
    
    def beam_raw(self, 
                 tokens: torch.Tensor, attn_mask: torch.Tensor, 
                 termination_condition: Callable[[np.ndarray], bool], 
                 beam_width=1, max_generation_len=None, 
                 prefix_embs: Optional[torch.Tensor]=None, 
                 prefix_attn_mask: Optional[torch.Tensor]=None, 
                 remove_prefix_position_embs: bool=False):
        tokenizer = self.bc_lm.dataset.tokenizer
        max_length = self.bc_lm.dataset.max_len
        if max_length is None:
            max_length = self.bc_lm.model.config.n_positions
        max_length = min(max_length, self.bc_lm.model.config.n_positions)
        device = self.bc_lm.device
        bsize, vocab_size = tokens.shape[0], tokenizer.num_tokens()
        n = bsize * beam_width
        if max_generation_len is None:
            max_generation_len = max_length+1
        input_strs = [tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.bc_lm(tokens, attn_mask, prefix_embs=prefix_embs, 
                                   prefix_attn_mask=prefix_attn_mask, 
                                   remove_prefix_position_embs=remove_prefix_position_embs, 
                                   use_cache=True)
        dialogue_kvs = model_outputs.past_key_values
        original_dialogue_lens = attn_mask.sum(dim=1)
        batch_indicator = torch.stack(beam_width*[torch.arange(0, bsize).to(device)], dim=1)
        tokens = pad_sequence(torch.repeat_interleave(tokens, beam_width, dim=0), max_length, tokenizer.pad_token_id, device, 1)
        dialogue_lens = torch.repeat_interleave(original_dialogue_lens, beam_width, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, beam_width, dim=0), max_length, 0.0, device, 2), dialogue_kvs)
        curr_scores = torch.zeros(bsize, beam_width).to(device)  # (batch, k)
        termination_mask = torch.full((n,), 1).to(device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            transformer_outputs = self.bc_lm(curr_token, None, past_key_values=curr_dialogue_kvs, use_cache=True)
            logits = transformer_outputs.logits
            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            scores = (torch.log(F.softmax(logits, dim=-1)).reshape(1, bsize, beam_width, -1).permute(3, 0, 1, 2) + curr_scores).permute(1, 2, 3, 0).reshape(1, bsize, -1)  # (time, batch, k*vocab)
            scores[0, :, vocab_size:] = scores[0, :, vocab_size:].masked_fill_((t == original_dialogue_lens).unsqueeze(1).repeat(1, scores.shape[2]-vocab_size), float('-inf'))
            curr_scores, top_k = torch.topk(scores[0, :, :], k=beam_width, dim=1)  # (batch, k), (batch, k)
            tokens = tokens[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :]
            tokens[:, t] = top_k.reshape(-1) % vocab_size  # (batch*k,)
            fixed_dialogue_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :, :, :], transformer_outputs.past_key_values)
            dialogue_kvs = map_all_kvs(lambda x: x[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1), :, :, :], dialogue_kvs)
            dialogue_kvs = update_kvs(dialogue_kvs, fixed_dialogue_kvs, torch.arange(0, n).to(device), (t+prefix_t)-1)
            dialogue_lens = dialogue_lens[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1)]
            termination_mask = termination_mask[(batch_indicator * beam_width + (top_k // vocab_size)).reshape(-1)]
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= (1 - int(termination_condition(tokenizer.decode(tokens[idx, :].tolist(),
                                                                                             clean_up_tokenization_spaces=False))))
            t += 1
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in range(n)]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(beam_width):
                processed_str = output_strs[i*beam_width+x][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        return list(zip(input_strs, processed_outputs)), curr_scores
    
    def generate(self, items: InputType, 
                 termination_condition: Callable[[np.ndarray], bool], **kwargs):
        prepared_inputs = self.bc_lm.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        if self.kind == 'beam':
            method = self.beam_raw
        elif self.kind == 'sample':
            method = self.sample_raw
        else:
            raise NotImplementedError
        generations, probs = method(tokens, attn_mask, 
                                    termination_condition, 
                                    **kwargs)
        return generations, probs
    
    def act(self, obs: Language_Observation) -> str:
        item = DataPoint.from_obs(obs, self.bc_lm.dataset.tokenizer, self.bc_lm.dataset.token_reward)
        generations, probs = self.generate([item], always_terminate, **self.generation_kwargs)
        sorted_outputs = list(zip(*sorted(zip(generations[0][1], probs[0]), key=lambda x: -x[1])))[0]
        return sorted_outputs[0]
    
    def train(self):
        self.bc_lm.train()
    
    def eval(self):
        self.bc_lm.eval()

class BC_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
    
    def evaluate(self, model: BC_LM, items: InputType) -> Optional[Dict[str, Any]]:
        policy = BC_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        n = tokens.shape[0]
        total_token_reward = 0
        total_env_reward = 0
        for i in range(n):
            result, sequence = interact_environment(self.env, policy, None)
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            total_env_reward += env_reward
            total_token_reward += token_reward
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('='*25)
        return {'token_reward': (total_token_reward / n, n), 'env_reward': (total_env_reward / n, n)}
