from typing import Any, Dict, List, Optional, Union
from data.language_environment import Language_Observation
from data.rl_data import DataPoint, RL_Dataset
from models.base import BaseModel, Evaluator, InputType
import torch
import numpy as np
from transformers.modeling_utils import PreTrainedModel
import torch.nn as nn
from models.gpt2_optional_final_ln import GPT2LMHeadModel, GPT2Model
from utils.sampling_utils import *
from data.language_environment import Language_Environment, Policy, interact_environment
import math
from utils.cache import Cache
from utils.torch_utils import map_pytree

class ChaiModel(BaseModel):
    def __init__(self, dataset: RL_Dataset, 
                 device: Union[torch.device, str], 
                 model: PreTrainedModel, 
                 alpha: float, 
                 gamma: float, 
                 generation_cache: Optional[Cache]):
        super().__init__(dataset, device)
        self.generation_cache = generation_cache
        model.resize_token_embeddings(self.dataset.tokenizer.num_tokens())
        self.lm_policy = model
        self.lm_policy.eval()
        self.h_dim  = self.lm_policy.config.n_embd
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(model, GPT2Model):
            self.pi = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim*2),
                nn.ReLU(), 
                nn.Linear(self.h_dim*2, self.dataset.tokenizer.num_tokens()),
            )
        else:
            self.pi = self.lm_policy.lm_head
        self.q = nn.Sequential(
                                nn.Linear(self.h_dim, self.h_dim*2), 
                                nn.ReLU(), 
                                nn.Linear(self.h_dim*2, 1), 
                              )
        self.q2 = nn.Sequential(
                                nn.Linear(self.h_dim, self.h_dim*2), 
                                nn.ReLU(), 
                                nn.Linear(self.h_dim*2, 1), 
                              )
        self.target_q = nn.Sequential(
                                nn.Linear(self.h_dim, self.h_dim*2), 
                                nn.ReLU(), 
                                nn.Linear(self.h_dim*2, 1), 
                              )
        self.target_q2 = nn.Sequential(
                                nn.Linear(self.h_dim, self.h_dim*2), 
                                nn.ReLU(), 
                                nn.Linear(self.h_dim*2, 1), 
                              )
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(local_param.data)
    
    def prepare_inputs(self, items: InputType):
        data = super().prepare_inputs(items)
        data['state_idxs'], data['action_idxs'] = data['u_state_idxs'], data['u_action_idxs']
        data['terminals'], data['rewards'] = data['u_terminals'], data['u_rewards']
        return data

    def qs_from_hs_state(self, h_state: torch.Tensor):
        qs = self.q(h_state).squeeze(2)
        qs2 = self.q2(h_state).squeeze(2)
        with torch.no_grad():
            target_qs = self.target_q(h_state).squeeze(2)
            target_qs2 = self.target_q2(h_state).squeeze(2)
        return ((qs, qs2,), torch.minimum(target_qs, target_qs2),)

    def forward(self, 
                tokens: torch.Tensor, 
                attn_mask: Optional[torch.Tensor], 
                state_idxs: torch.Tensor, 
                action_idxs: torch.Tensor, 
                prefix_embs: Optional[torch.Tensor]=None, 
                prefix_attn_mask: Optional[torch.Tensor]=None, 
                remove_prefix_position_embs: bool=False, 
                **kwargs):
        if attn_mask is None:
            attn_mask = torch.ones(tokens.shape, dtype=torch.long).to(self.device)
        if prefix_embs is None:
            prefix_embs = torch.empty((tokens.shape[0], 0, self.h_dim)).to(self.device)
        prefix_t = prefix_embs.shape[1]
        set_pos_ids = prefix_attn_mask is not None
        if prefix_attn_mask is None:
            prefix_attn_mask = torch.ones(prefix_embs.shape[:2]).to(self.device)
        input_attn_mask = torch.cat((prefix_attn_mask, attn_mask), dim=1)
        position_ids = torch.cumsum(input_attn_mask, dim=1)-1 if set_pos_ids else None
        if isinstance(self.lm_policy, GPT2Model):
            transformer = self.lm_policy
        elif isinstance(self.lm_policy, GPT2LMHeadModel):
            transformer = self.lm_policy.transformer
        else:
            raise NotImplementedError
        if remove_prefix_position_embs:
            prefix_embs -= transformer.wpe(position_ids[:, :prefix_embs.shape[1]])
        input_embeddings = torch.cat((prefix_embs, transformer.wte(tokens)), dim=1)
        with torch.no_grad():
            model_outputs = self.lm_policy(inputs_embeds=input_embeddings, 
                                           attention_mask=input_attn_mask, 
                                           position_ids=position_ids, 
                                           output_hidden_states=True, 
                                           **kwargs)
            raw_h_states = model_outputs.hidden_states[-1][:, prefix_t:, :].detach()
            hidden_states = (torch.cumsum(raw_h_states, dim=1) / (torch.arange(0, raw_h_states.shape[1])+1).to(self.device).unsqueeze(1)).detach()
            action_hidden_states = torch.gather(input=hidden_states, dim=1, index=action_idxs.unsqueeze(2).repeat(1, 1, self.h_dim))
        (qs, qs2), target_qs = self.qs_from_hs_state(action_hidden_states)
        with torch.no_grad():
            logits = self.pi(raw_h_states)
        return  {
                    'model_outputs': model_outputs, 
                    'qs': (qs, qs2,), 
                    'target_qs': target_qs, 
                    'action_hidden_states': action_hidden_states, 
                    'logits': logits, 
                }

    def produce_samples(self, 
                        tokens: torch.Tensor, 
                        attn_mask: Optional[torch.Tensor], 
                        state_idxs: torch.Tensor, 
                        action_idxs: torch.Tensor, 
                        n_generations: int, 
                        max_generation_len=None, 
                        temp=1.0, top_k=None, top_p=None, 
                        bsize=1):
        bsize = tokens.shape[0]
        if action_idxs.shape[1] == 0:
            n = torch.zeros((bsize,)).long().to(self.device)
        else:
            n = torch.argmax(action_idxs, dim=1)+1
        to_query = {}
        results = {}
        for i in range(bsize):
            for t in range(n[i].item()):
                t_idx = state_idxs[i, t].item()
                prefix_str = self.dataset.tokenizer.decode(tokens[i, :(t_idx+1)], clean_up_tokenization_spaces=False)
                if self.generation_cache is not None and prefix_str in self.generation_cache:
                    io_strs, log_probs, h_states = self.generation_cache[prefix_str]
                    
                    # mo_1 = self(tokens[i, :t_idx].unsqueeze(0), None, torch.full((1,1,), 0).long().to(self.device), torch.full((1,1,), t_idx-1).long().to(self.device), use_cache=True)
                    # p_kv1 = mo_1['model_outputs'].past_key_values
                    # p_kv1 = map_all_kvs(lambda x: torch.repeat_interleave(x, n_generations, dim=0), p_kv1)
                    # new_tokens, new_attn = self.dataset.tokenizer.encode(list(map(lambda x: '</s> '+x+' </a>', io_strs[0][1])))
                    # new_tokens, new_attn = torch.tensor(new_tokens).to(self.device), torch.tensor(new_attn).to(self.device)
                    # mo_2 = self(new_tokens, torch.ones((n_generations,t_idx+new_tokens.shape[1])).to(self.device), torch.full((n_generations,1,), 0).long().to(self.device), (new_attn.sum(dim=1)-1).unsqueeze(1), use_cache=True, past_key_values=p_kv1)
                    # mo_2['action_hidden_states'] = (((mo_1['action_hidden_states'][0, -1, :] * t_idx) + (mo_2['action_hidden_states'][:, -1, :] * (new_attn.sum(dim=1).unsqueeze(1)))) / (t_idx+new_attn.sum(dim=1).unsqueeze(1))).unsqueeze(1)
                    # h_states = mo_2['action_hidden_states'].unsqueeze(0)
                    # self.generation_cache[prefix_str] = (io_strs, log_probs.detach().cpu(), h_states.detach().cpu(),)

                    log_probs, h_states = log_probs.to(self.device), h_states.to(self.device)
                    (q1, q2), target_qs = self.qs_from_hs_state(h_states)
                    q1, q2, target_qs = q1.squeeze(2), q2.squeeze(2), target_qs.squeeze(2)
                    results[(i, t,)] = io_strs, log_probs, ((q1, q2,), target_qs,), h_states
                else:
                    to_query[((i, t,), prefix_str,)] = (tokens[i, :(t_idx+1)].unsqueeze(0), 
                                                        attn_mask[i, :(t_idx+1)].unsqueeze(0), 
                                                        state_idxs[i, :(t+1)].unsqueeze(0), 
                                                        action_idxs[i, :t].unsqueeze(0),)
                    if len(to_query) > bsize:
                        k, v = zip(*to_query.items())
                        new_tokens, new_attn_mask, new_state_idxs, new_action_idxs = list(zip(*v))
                        new_tokens = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_tokens)), batch_first=True, padding_value=self.dataset.tokenizer.pad_token_id)
                        new_attn_mask = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_attn_mask)), batch_first=True, padding_value=0)
                        new_state_idxs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_state_idxs)), batch_first=True, padding_value=0)
                        new_action_idxs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_action_idxs)), batch_first=True, padding_value=0)
                        io_strs, log_probs, ((q1, q2,), target_qs,), h_states = self.sample_raw(new_tokens, new_attn_mask, 
                                                                                                new_state_idxs, 
                                                                                                new_action_idxs, n_generations, 
                                                                                                max_generation_len, 
                                                                                                temp, top_k, top_p)
                        for x, ((o_i, o_t), o_prefix_str,) in enumerate(k):
                            if self.generation_cache is not None:
                                self.generation_cache[o_prefix_str] = ([io_strs[x]], log_probs[x].unsqueeze(0).detach().cpu(), h_states[x].unsqueeze(0).detach().cpu(),)
                            results[(o_i, o_t,)] = ([io_strs[x]], log_probs[x].unsqueeze(0), ((q1[x].unsqueeze(0), q2[x].unsqueeze(0),), target_qs[x].unsqueeze(0),), h_states[x].unsqueeze(0),)
                        to_query = {}
        if len(to_query) > 0:
            k, v = zip(*to_query.items())
            new_tokens, new_attn_mask, new_state_idxs, new_action_idxs = list(zip(*v))
            new_tokens = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_tokens)), batch_first=True, padding_value=self.dataset.tokenizer.pad_token_id)
            new_attn_mask = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_attn_mask)), batch_first=True, padding_value=0)
            new_state_idxs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_state_idxs)), batch_first=True, padding_value=0)
            new_action_idxs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x.squeeze(0), new_action_idxs)), batch_first=True, padding_value=0)
            io_strs, log_probs, ((q1, q2,), target_qs,), h_states = self.sample_raw(new_tokens, new_attn_mask, 
                                                                                    new_state_idxs, 
                                                                                    new_action_idxs, n_generations, 
                                                                                    max_generation_len, 
                                                                                    temp, top_k, top_p)
            for x, ((o_i, o_t), o_prefix_str,) in enumerate(k):
                if self.generation_cache is not None:
                    self.generation_cache[o_prefix_str] = ([io_strs[x]], log_probs[x].unsqueeze(0).detach().cpu(), h_states[x].unsqueeze(0).detach().cpu(),)
                results[(o_i, o_t,)] = ([io_strs[x]], log_probs[x].unsqueeze(0), ((q1[x].unsqueeze(0), q2[x].unsqueeze(0),), target_qs[x].unsqueeze(0),), h_states[x].unsqueeze(0),)
            to_query = {}
        all_raw_strs = []
        all_log_probs = []
        all_q1 = []
        all_q2 = []
        all_target_qs = []
        all_h_states = []
        for i in range(bsize):
            temp_raw_strs = []
            temp_log_probs = []
            temp_q1 = []
            temp_q2 = []
            temp_target_qs = []
            temp_h_states = []
            for t in range(n[i].item()):
                io_strs, log_probs, ((q1, q2,), target_qs,), h_states = results[(i, t,)]
                temp_raw_strs.append(io_strs[0])
                temp_log_probs.append(log_probs[0])
                temp_q1.append(q1[0])
                temp_q2.append(q2[0])
                temp_target_qs.append(target_qs[0])
                temp_h_states.append(h_states[0, :, -1, :])
            all_raw_strs.append(temp_raw_strs)
            all_log_probs.append(pad_sequence(torch.stack(temp_log_probs, dim=0), state_idxs.shape[1], 0.0, self.device, 0))
            all_q1.append(pad_sequence(torch.stack(temp_q1, dim=0), state_idxs.shape[1], 0.0, self.device, 0))
            all_q2.append(pad_sequence(torch.stack(temp_q2, dim=0), state_idxs.shape[1], 0.0, self.device, 0))
            all_target_qs.append(pad_sequence(torch.stack(temp_target_qs, dim=0), state_idxs.shape[1], 0.0, self.device, 0))
            all_h_states.append(pad_sequence(torch.stack(temp_h_states, dim=0), state_idxs.shape[1], 0.0, self.device, 0))
        all_log_probs = torch.stack(all_log_probs, dim=0)
        all_q1 = torch.stack(all_q1, dim=0)
        all_q2 = torch.stack(all_q2, dim=0)
        all_target_qs = torch.stack(all_target_qs, dim=0)
        all_h_states = torch.stack(all_h_states, dim=0)
        return all_raw_strs, all_log_probs, ((all_q1, all_q2,), all_target_qs,), all_h_states

    def random_q_values(self, bsize, sequence_length, n_generations):
        embs = (2*torch.rand(bsize, sequence_length, n_generations, self.h_dim)-1).to(self.device)
        q1 = self.q(embs).squeeze(-1)
        q2 = self.q2(embs).squeeze(-1)
        target_qs = torch.minimum(self.target_q(embs).squeeze(-1), self.target_q2(embs).squeeze(-1))
        log_probs = torch.full((bsize, sequence_length, n_generations,), -math.log(2*self.h_dim)).to(self.device)
        return ((q1, q2,), target_qs,), log_probs

    def get_q_loss(self, qs, sample_next_target_qs, rs, gamma, terminals):
        sample_next_target_qs = sample_next_target_qs.detach()
        max_qns = torch.max(sample_next_target_qs, dim=-1).values
        q1, q2 = qs
        l1 = ((((1 - terminals[:, 1:]) * max_qns * gamma + rs - q1) ** 2) * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
        l2 = ((((1 - terminals[:, 1:]) * max_qns * gamma + rs - q2) ** 2) * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
        return l1 + l2

    def get_cql_loss(self, qs, sample_qs, sample_log_probs, random_qs, random_log_probs, terminals):
        q1, q2 = qs
        random_q1, random_q2 = random_qs
        sample_q1, sample_q2 = sample_qs
        n_generations = sample_q1.shape[-1]
        logsumexp1 = torch.logsumexp((torch.cat((sample_q1 - sample_log_probs, random_q1 - random_log_probs), dim=2) / (2 * n_generations)), dim=2)
        logsumexp2 = torch.logsumexp((torch.cat((sample_q2 - sample_log_probs, random_q2 - random_log_probs), dim=2) / (2 * n_generations)), dim=2)
        cql_loss = (((logsumexp1 - q1) + (logsumexp2 - q2)) * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
        return cql_loss

    def sample_raw(self, 
                   tokens: torch.Tensor, attn_mask: torch.Tensor, 
                   state_idxs: torch.Tensor, action_idxs: torch.Tensor, 
                   num_generations=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None, 
                   prefix_embs: Optional[torch.Tensor]=None, 
                   prefix_attn_mask: Optional[torch.Tensor]=None, 
                   remove_prefix_position_embs: bool=False):
        tokenizer = self.dataset.tokenizer
        max_length = self.dataset.max_len
        if max_length is None:
            max_length = self.lm_policy.config.n_positions
        max_length = min(max_length, self.lm_policy.config.n_positions)
        device = self.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length+1
        input_strs = [tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        dialogue_lens = attn_mask.sum(dim=1)
        state_idxs_temp, action_idxs_temp = ((dialogue_lens-1).unsqueeze(1).long(), (dialogue_lens-1).unsqueeze(1).long())
        all_outputs = self(tokens, attn_mask, 
                           state_idxs=state_idxs_temp, 
                           action_idxs=action_idxs_temp, 
                           prefix_embs=prefix_embs, 
                           prefix_attn_mask=prefix_attn_mask, 
                           remove_prefix_position_embs=remove_prefix_position_embs, 
                           use_cache=True)
        model_outputs = all_outputs['model_outputs']
        qs, target_qs, action_hidden_states = all_outputs['qs'], all_outputs['target_qs'], all_outputs['action_hidden_states']
        qs = [torch.repeat_interleave(qs[0], num_generations, dim=0), torch.repeat_interleave(qs[1], num_generations, dim=0)]
        target_qs = torch.repeat_interleave(target_qs, num_generations, dim=0)
        action_hidden_states = torch.repeat_interleave(action_hidden_states, num_generations, dim=0)
        kvs = model_outputs.past_key_values
        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), max_length, tokenizer.pad_token_id, device, 1)
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2), kvs)
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        state_idxs_temp, action_idxs_temp = torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device), torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device)
        t = torch.min(dialogue_lens).int()
        action_hidden_states = action_hidden_states[:, -1, :] * dialogue_lens.unsqueeze(1)
        total_dialogue_lens = dialogue_lens.clone()
        while termination_mask.sum() > 0 and (t+prefix_t) < max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], kvs)
            all_outputs = self(curr_token, None, state_idxs_temp, action_idxs_temp, 
                               use_cache=True, past_key_values=curr_kvs)
            model_outputs, logits = all_outputs['model_outputs'], all_outputs['logits']
            condition = (termination_mask.unsqueeze(1) + (curr_token == tokenizer.eoa_token_id).int()) * (t > dialogue_lens).int().unsqueeze(1)
            action_hidden_states += all_outputs['action_hidden_states'][:, -1, :] * condition

            (qs, target_qs) = self.qs_from_hs_state(action_hidden_states.unsqueeze(1))
            # qs[0] = all_outputs['qs'][0] * condition + qs[0] * (1 - condition)
            # qs[1] = all_outputs['qs'][1] * condition + qs[1] * (1 - condition)
            # target_qs = all_outputs['target_qs'] * condition + target_qs * (1 - condition)

            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            edited_logits = process_logits(logits.clone(), temp=temp, top_k=top_k, top_p=top_p)
            cat_dist = torch.distributions.categorical.Categorical(logits=edited_logits[:, 0])
            new_tokens = cat_dist.sample()
            log_probs += cat_dist.log_prob(new_tokens)
            tokens[:, t] = new_tokens
            kvs = update_kvs(kvs, model_outputs.past_key_values, torch.arange(0, n).to(device), (t+prefix_t)-1)
            total_dialogue_lens += termination_mask * (t >= dialogue_lens).int()
            for idx in range(n):
                if tokens[idx, t] == tokenizer.eoa_token_id and t >= dialogue_lens[idx]:
                    termination_mask[idx] *= 0
            t += 1
            termination_mask *= ((t-dialogue_lens) < max_generation_len).int()
        curr_token = tokens[:, t-1].unsqueeze(1)
        curr_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], kvs)
        all_outputs = self(curr_token, None, state_idxs_temp, action_idxs_temp, 
                            use_cache=True, past_key_values=curr_kvs)
        condition = (termination_mask.unsqueeze(1) + (curr_token == tokenizer.eoa_token_id).int()) * (t > dialogue_lens).int().unsqueeze(1)
        action_hidden_states += all_outputs['action_hidden_states'][:, -1, :] * condition

        (qs, target_qs) = self.qs_from_hs_state(action_hidden_states.unsqueeze(1))
        # qs[0] = all_outputs['qs'][0] * condition + qs[0] * (1 - condition)
        # qs[1] = all_outputs['qs'][1] * condition + qs[1] * (1 - condition)
        # target_qs = all_outputs['target_qs'] * condition + target_qs * (1 - condition)
        
        scores = target_qs.reshape(bsize, num_generations)
        order = torch.argsort(-scores, dim=1)
        output_strs = [tokenizer.decode(tokens[i, :].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        processed_outputs = []
        for i in range(len(input_strs)):
            temp_outputs = []
            for x in range(num_generations):
                processed_str = output_strs[i*num_generations+order[i, x]][len(input_strs[i]):].strip()
                if tokenizer.id_to_token(tokenizer.pad_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.pad_token_id))].strip()
                if tokenizer.id_to_token(tokenizer.eoa_token_id) in processed_str:
                    processed_str = processed_str[:processed_str.find(tokenizer.id_to_token(tokenizer.eoa_token_id))].strip()
                temp_outputs.append(processed_str)
            processed_outputs.append(temp_outputs)
        scores = torch.gather(scores, dim=1, index=order)
        log_probs = torch.gather(log_probs.reshape(bsize, num_generations), dim=1, index=order)
        qs = [qs[0], qs[1]]
        qs[0] = torch.gather(qs[0].reshape(bsize, num_generations), dim=1, index=order)
        qs[1] = torch.gather(qs[1].reshape(bsize, num_generations), dim=1, index=order)
        target_qs = torch.gather(target_qs.reshape(bsize, num_generations), dim=1, index=order)
        action_hidden_states = action_hidden_states / total_dialogue_lens.unsqueeze(1)
        action_hidden_states = torch.gather(action_hidden_states.reshape(bsize, num_generations, self.h_dim), dim=1, index=order.unsqueeze(2).repeat(1, 1, self.h_dim))
        return list(zip(input_strs, processed_outputs)), log_probs, ((qs[0], qs[1],), target_qs,), action_hidden_states.unsqueeze(2)

    def get_loss(self, 
                 items: InputType, 
                 q_loss_weight=0.0, 
                 v_loss_weight=0.0, 
                 cql_loss_weight=0.0, 
                 n_generations=1, 
                 max_generation_len=None, 
                 temp=1.0, top_k=None, top_p=None, 
                 sample_bsize=1):
        prepared_inputs = self.prepare_inputs(items)
        s_idx, a_idx, rs = prepared_inputs['state_idxs'], prepared_inputs['action_idxs'], prepared_inputs['rewards']
        tokens, attn_mask, terminals = prepared_inputs['tokens'], prepared_inputs['attn_mask'], prepared_inputs['terminals']
        self_outputs = self(tokens, attn_mask, s_idx, a_idx)
        _, sample_log_probs, (sample_qs, sample_target_qs,), _ = self.produce_samples(tokens, attn_mask, s_idx, a_idx, n_generations, max_generation_len, temp, top_k, top_p, sample_bsize)
        (random_qs, _,), random_log_probs = self.random_q_values(a_idx.shape[0], a_idx.shape[1], n_generations)
        q_loss = self.get_q_loss(self_outputs['qs'], sample_target_qs[:, 1:], rs, self.gamma, terminals)
        cql_loss = self.get_cql_loss(self_outputs['qs'], (sample_qs[0][:, :-1], sample_qs[1][:, :-1],), sample_log_probs[:, :-1], random_qs, random_log_probs, terminals)
        logs = {}
        n = (1 - terminals[:, :-1]).sum().item()
        loss = q_loss_weight * q_loss + cql_loss_weight * cql_loss
        logs['q_loss'] = (q_loss.item(), n)
        logs['cql_loss'] = (cql_loss.item(), n)
        q1, q2 = self_outputs['qs']
        logs['q1_avg'] = ((q1 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
        logs['q1_var'] = (((((q1 - logs['q1_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        logs['q2_avg'] = ((q2 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
        logs['q2_var'] = (((((q2 - logs['q2_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        postproc_f = lambda l: l.update({'loss': q_loss_weight * l['q_loss'] + cql_loss_weight * l['cql_loss']})
        return loss, logs, [postproc_f]
    
    def soft_update(self):
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(self.alpha*local_param.data + (1.0-self.alpha)*target_param.data)
        for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.alpha*local_param.data + (1.0-self.alpha)*target_param.data)
    
    def hard_update(self):
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(local_param.data)
        for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(local_param.data)
    
    def train(self, mode=True):
        r_val = super().train(mode)
        self.lm_policy.eval()
        return r_val
    
    def eval(self):
        return super().eval()

class ChaiPolicy(Policy):
    def __init__(self, 
                 chai_model: ChaiModel, 
                 **generation_kwargs):
        super().__init__()
        self.chai_model = chai_model
        self.generation_kwargs = generation_kwargs
    
    def act(self, obs: Language_Observation) -> str:
        curr_kwargs = dict(self.generation_kwargs)
        best_score = float('-inf')
        selected_items = None
        for i in range(curr_kwargs.pop('generation_batches')):
            item = DataPoint.from_obs(obs, self.chai_model.dataset.tokenizer, self.chai_model.dataset.token_reward)
            prepared_inputs = self.chai_model.prepare_inputs([item])
            tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
            state_idxs, action_idxs = prepared_inputs['state_idxs'], prepared_inputs['action_idxs']
            samples, _, (_, scores), _ = self.chai_model.sample_raw(tokens, attn_mask, state_idxs, action_idxs, **curr_kwargs)
            # print(samples, scores)
            if scores[0, 0].item() > best_score:
                best_score = scores[0, 0].item()
                selected_items = (samples, scores)
        (generations, scores) = selected_items
        return generations[0][1][0]

    def train(self):
        self.chai_model.train()

    def eval(self):
        self.chai_model.eval()

class Chai_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, cache_save_path: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.cache_save_path = cache_save_path
        self.generation_kwargs = generation_kwargs
    
    def evaluate(self, model: ChaiModel, items: InputType) -> Optional[Dict[str, Any]]:
        if self.cache_save_path is not None:
            if self.verbose:
                print('dumping cache to %s' % (self.cache_save_path))
            model.generation_cache.dump(self.cache_save_path)
            if self.verbose:
                print('dumped.')
        policy = ChaiPolicy(model, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        total_token_reward = 0
        total_env_reward = 0
        for i in range(tokens.shape[0]):
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
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0])}

