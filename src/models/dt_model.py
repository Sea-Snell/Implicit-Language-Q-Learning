from data.language_environment import Language_Environment, interact_environment
from models.base import Evaluator, InputType
from models.bc_lm import BC_LM, BC_Policy
from transformers.modeling_utils import PreTrainedModel
from data.rl_data import DataPoint, RL_Dataset
from typing import Any, Callable, Dict, Optional, Union
import torch
import torch.nn as nn
import numpy as np

class DT(BC_LM):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 transition_weight: float=0.0, 
                ):
        super().__init__(model, dataset, device, transition_weight=transition_weight)
        self.reward_emb = nn.Linear(1, self.h_dim)

    def get_loss(self, 
                 items: InputType):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        a_idx = prepared_inputs['action_idxs']
        prefix_embs = self.reward_emb(prepared_inputs['rewards'].sum(dim=1).unsqueeze(1)).unsqueeze(1)
        model_outputs = self(tokens, attn_mask, 
                             prefix_embs=prefix_embs, 
                             output_attentions=True)
        logs = {}
        n = attn_mask.sum().item()
        weights = self.get_weights(tokens, a_idx)
        token_loss = self.awac_loss(tokens, attn_mask, model_outputs.logits[:, 1:, :], weights)
        logs['loss'] = (token_loss.item(), n)
        return token_loss, logs, []

class DT_Policy(BC_Policy):    
    def generate(self, items: InputType, 
                 termination_condition: Callable[[np.ndarray], bool], **kwargs):
        kwargs = dict(kwargs)
        cond_r = kwargs.pop('cond_r')
        prepared_inputs = self.bc_lm.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        prefix_embs = self.bc_lm.reward_emb(torch.full((tokens.shape[0],1,), cond_r).to(self.bc_lm.device)).unsqueeze(1)
        if self.kind == 'beam':
            method = self.beam_raw
        elif self.kind == 'sample':
            method = self.sample_raw
        else:
            raise NotImplementedError
        generations, probs = method(tokens, attn_mask, 
                                    termination_condition, 
                                    prefix_embs=prefix_embs, 
                                    **kwargs)
        return generations, probs

class DT_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
    
    def evaluate(self, model: BC_LM, items: InputType) -> Optional[Dict[str, Any]]:
        policy = DT_Policy(model, self.kind, **self.generation_kwargs)
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
