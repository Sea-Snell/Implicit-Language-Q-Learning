import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Tuple, Union, Optional
from data.language_environment import Language_Environment, Language_Observation, interact_environment
from data.rl_data import DataPoint, RL_Dataset
from models.iql_model import IQL_Policy, PerTokenIQL, TransformerMLP
from models.base import Evaluator, InputType
from transformers.modeling_utils import PreTrainedModel
from utils.sampling_utils import *
from utils.torch_utils import get_transformer_logs
import wandb
import math

class PerUtteranceIQL(PerTokenIQL):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 alpha: float = 0.005, 
                 gamma=1.0, 
                 beta=1.0, 
                 transition_weight=0.0, 
                 clip_weight: Optional[float] = None, 
                 value_max: Optional[float] = None, 
                 value_min: Optional[float] = None, 
                 detach_v: bool = False, 
                 detach_pi: bool = False, 
                 detach_q: bool = False, 
                 double_q: bool = False, 
                 tau: float = 0.9, 
                 seperate_policy: bool = False, 
                 seperate_target: bool = False, 
                 exp_weights: bool = False, 
                 advanced_mlp: bool = False, 
                ):
        super(PerUtteranceIQL, self).__init__(model, dataset, device, alpha, 
                                              gamma, beta, transition_weight, clip_weight, 
                                              value_max, value_min, detach_v, 
                                              detach_pi, detach_q, double_q, tau, 
                                              seperate_policy, seperate_target, 
                                              exp_weights, 0.0, advanced_mlp, 1.0)
        if not self.advanced_mlp:
            self.q = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim*2),
                nn.ReLU(), 
                nn.Linear(self.h_dim*2, 1),
            )
        else:
            self.q = TransformerMLP(self.h_dim, 
                                    4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner, 
                                    1, self.model.config.resid_pdrop)
        if self.double_q:
            if not self.advanced_mlp:
                self.q2 = nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim*2),
                    nn.ReLU(), 
                    nn.Linear(self.h_dim*2, 1),
                )
            else:
                self.q2 = TransformerMLP(self.h_dim, 
                                         4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner, 
                                         1, self.model.config.resid_pdrop)
        if not self.advanced_mlp:
            self.target_q = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim*2),
                nn.ReLU(), 
                nn.Linear(self.h_dim*2, 1),
            )
        else:
            self.target_q = TransformerMLP(self.h_dim, 
                                           4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner, 
                                           1, self.model.config.resid_pdrop)
        if self.double_q:
            if not self.advanced_mlp:
                self.target_q2 = nn.Sequential(
                    nn.Linear(self.h_dim, self.h_dim*2),
                    nn.ReLU(), 
                    nn.Linear(self.h_dim*2, 1),
                )
            else:
                self.target_q2 = TransformerMLP(self.h_dim, 
                                                4 * self.h_dim if self.model.config.n_inner is None else self.model.config.n_inner, 
                                                1, self.model.config.resid_pdrop)
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(local_param.data)
        if self.double_q:
            for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(local_param.data)
    
    def prepare_inputs(self, items: InputType):
        data = super().prepare_inputs(items)
        data['state_idxs'], data['action_idxs'] = data['u_state_idxs'], data['u_action_idxs']
        data['terminals'], data['rewards'] = data['u_terminals'], data['u_rewards']
        return data
    
    def get_weights(self, 
                    tokens: torch.Tensor, 
                    vs: torch.Tensor, 
                    qs: Optional[torch.Tensor], 
                    state_idxs: torch.Tensor, 
                    action_idxs: torch.Tensor, 
                    terminals: torch.Tensor):
        weights = torch.full(tokens.shape, self.transition_weight).to(self.device)
        if self.exp_weights:
            w_values = torch.exp(self.beta * (qs - vs))
        else:
            # w_values = ((qs - vs) > 0.0).float()
            adv_sign = ((qs - vs) > 0.0).float()
            w_values = self.beta * adv_sign + (1 - self.beta) * (1 - adv_sign)
        n = torch.argmax(action_idxs, dim=1)+1
        for i in range(tokens.shape[0]):
            for x in range(n[i].item()):
                weights[i] = torch.scatter(weights[i], dim=0, 
                                           index=torch.arange(state_idxs[i, x], action_idxs[i, x]).to(self.device), 
                                           src=w_values[i, x].repeat(action_idxs[i, x]-state_idxs[i, x]))
        if self.clip_weight is not None:
            weights = torch.clip(weights, max=self.clip_weight)
        # print(list(map(lambda x: list(map(lambda y: (y[0], self.dataset.tokenizer.id_to_token(y[1].item()),), zip(*x))), zip(weights.detach().cpu().tolist(), tokens))))
        return weights
    
    def get_qvs(self, items: InputType, 
                prefix_embs: Optional[torch.Tensor]=None, 
                prefix_attn_mask: Optional[torch.Tensor]=None, 
                remove_prefix_position_embs: bool=False, 
                qv_kwargs=None, policy_kwargs=None, target_kwargs=None, 
                **kwargs):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        s_idx, a_idx = prepared_inputs['state_idxs'], prepared_inputs['action_idxs']
        rs, terminals = prepared_inputs['rewards'], prepared_inputs['terminals']
        self_outputs = self(tokens, attn_mask, s_idx, a_idx, 
                            prefix_embs, prefix_attn_mask, 
                            remove_prefix_position_embs, qv_kwargs, 
                            policy_kwargs, target_kwargs, 
                            **kwargs)
        model_outputs, vs, qs = self_outputs['model_outputs'], self_outputs['vs'], self_outputs['qs']
        target_qs, logits = self_outputs['target_qs'], self_outputs['logits']
        vt = vs[:, :-1]
        vtp1 = vs[:, 1:]
        if self.double_q:
            q1, q2 = qs
            q1, q2 = q1.squeeze(2), q2.squeeze(2)
            qs = (q1, q2,)
        else:
            qs = qs.squeeze(2)
        target_qs = target_qs.squeeze(2)
        with torch.no_grad():
            weights = self.get_weights(tokens, vt, target_qs, s_idx, a_idx, terminals)
        return {
                    'tokens': tokens, 
                    'attn_mask': attn_mask, 
                    'model_outputs': model_outputs, 
                    'vs': vt, 
                    'qs': qs, 
                    'vns': vtp1, 
                    'target_vs': vt, 
                    'target_qs': target_qs, 
                    'target_vns': vtp1, 
                    'rs': rs, 
                    'terminals': terminals, 
                    'logits': logits, 
                    'weights': weights, 
                }
    
    def get_loss(self, 
                 items: InputType, 
                 awac_weight=0.0, 
                 v_loss_weight=0.0, 
                 q_loss_weight=0.0, 
                 mc_returns=False):
        prepared_inputs = self.prepare_inputs(items)
        a_idx = prepared_inputs['action_idxs']

        get_qvs_outputs = self.get_qvs(items, 
                                       qv_kwargs={'output_attentions': True}, 
                                       policy_kwargs={'output_attentions': True}, 
                                       target_kwargs={'output_attentions': True}, 
                                       skip_policy_on_train=(awac_weight == 0.0), 
                                      )
        tokens, attn_mask, model_outputs = get_qvs_outputs['tokens'], get_qvs_outputs['attn_mask'], get_qvs_outputs['model_outputs']
        vs, qs = get_qvs_outputs['vs'], get_qvs_outputs['qs']
        vns, target_qs, rs = get_qvs_outputs['vns'], get_qvs_outputs['target_qs'], get_qvs_outputs['rs']
        terminals, logits, weights = get_qvs_outputs['terminals'], get_qvs_outputs['logits'], get_qvs_outputs['weights']
        
        logs = {}
        transformer_logs = {}
        transformer_logs['qv_transformer_logs'] = get_transformer_logs(model_outputs['qv_model_outputs'].attentions, self.model, attn_mask)
        if self.lm_policy is not None and (not (self.training and awac_weight == 0.0)):
            transformer_logs['policy_transformer_logs'] = get_transformer_logs(model_outputs['policy_model_outputs'].attentions, self.lm_policy, attn_mask)
        if self.lm_target is not None:
            transformer_logs['target_transformer_logs'] = get_transformer_logs(model_outputs['target_model_outputs'].attentions, self.lm_target, attn_mask)
        n = (1 - terminals[:, :-1]).sum().item()
        rs_downstream = self.get_downstream_rs(rs, self.gamma)
        if mc_returns:
            v_loss = self.get_v_loss(vs, rs_downstream, terminals)
        else:
            v_loss = self.get_v_loss(vs, target_qs, terminals)
        q_loss = self.get_q_loss(vns, qs, rs, self.gamma, terminals)
        token_loss = self.awac_loss(tokens, attn_mask, logits, weights)
        logs['token_loss'] = (token_loss.item(), n)
        loss = awac_weight * token_loss + v_loss_weight * v_loss + q_loss_weight * q_loss
        logs['v_loss'] = (v_loss.item(), n)
        logs['q_loss'] = (q_loss.item(), n)
        advantages = sum([((target_qs[i] - vs[i])[:(1 - terminals[i, :-1]).sum().long().item()]).detach().cpu().tolist() for i in range(tokens.shape[0])], [])
        if self.double_q:
            q1, q2 = qs
            logs['q1_avg'] = ((q1 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q1_var'] = (((((q1 - logs['q1_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
            logs['q2_avg'] = ((q2 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q2_var'] = (((((q2 - logs['q2_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        else:
            logs['q_avg'] = ((qs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q_var'] = (((((qs - logs['q_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        logs['v_avg'] = ((vs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
        logs['v_var'] = (((((vs - logs['v_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        act_weights = torch.gather(weights, dim=1, index=torch.maximum(a_idx-1, torch.tensor(0).to(self.device)))
        logs['act_weight_avg'] = (((act_weights * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), n)
        logs['transformer'] = transformer_logs
        postproc_f = lambda l: l.update({'loss': awac_weight * l['token_loss'] + q_loss_weight * l['q_loss'] + v_loss_weight * l['v_loss']})
        hist_f = lambda l: l.update({'advantage_hist': wandb.Histogram(advantages)})
        return loss, logs, [postproc_f, hist_f]
    
    def get_scores(self, items: InputType, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def initial_score(self, items: InputType, **kwargs) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError
    
    def next_score(self, tokens: torch.Tensor, state: Any, **kwargs) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

class PerUtteranceIQL_Policy(IQL_Policy):
    def __init__(self, iql_model: PerTokenIQL, 
                 kind: str, **generation_kwargs) -> None:
        super().__init__(iql_model, 'sample', **generation_kwargs)
        assert kind in {'rerank'}
        self.kind = kind
    
    def rerank_raw(self, 
                   tokens: torch.Tensor, attn_mask: torch.Tensor, 
                   state_idxs: torch.Tensor, action_idxs: torch.Tensor, 
                   termination_condition: Callable[[np.ndarray], bool], 
                   num_generations=1, max_generation_len=None, 
                   temp=1.0, top_k=None, top_p=None, 
                   log_prob_weight=0.0, 
                   prefix_embs: Optional[torch.Tensor]=None, 
                   prefix_attn_mask: Optional[torch.Tensor]=None, 
                   remove_prefix_position_embs: bool=False):
        
        # swap out models so that only the relevent model is executed for speed purposes.
        temp_target = self.iql_model.lm_target
        temp_policy = self.iql_model.lm_policy
        temp_model = self.iql_model.model

        self.iql_model.lm_target = None
        self.iql_model.lm_policy = None
        self.iql_model.model = temp_policy

        tokenizer = self.iql_model.dataset.tokenizer
        max_length = self.iql_model.dataset.max_len
        if max_length is None:
            max_length = self.iql_model.model.config.n_positions
        max_length = min(max_length, self.iql_model.model.config.n_positions)
        device = self.iql_model.device
        bsize = tokens.shape[0]
        n = bsize * num_generations
        if max_generation_len is None:
            max_generation_len = max_length+1
        input_strs = [tokenizer.decode(tokens[i, :][:attn_mask[i, :].sum().long()].tolist(), clean_up_tokenization_spaces=False) for i in range(len(tokens))]
        prefix_t = 0 if prefix_embs is None else prefix_embs.shape[1]
        model_outputs = self.iql_model(tokens, attn_mask, 
                                       state_idxs, action_idxs, 
                                       prefix_embs=prefix_embs, 
                                       prefix_attn_mask=prefix_attn_mask, 
                                       remove_prefix_position_embs=remove_prefix_position_embs, 
                                       policy_kwargs={'use_cache': True})['model_outputs']['policy_model_outputs']
        dialogue_kvs = model_outputs.past_key_values
        dialogue_lens = attn_mask.sum(dim=1)
        tokens = pad_sequence(torch.repeat_interleave(tokens, num_generations, dim=0), max_length, tokenizer.pad_token_id, device, 1)
        dialogue_lens = torch.repeat_interleave(dialogue_lens, num_generations, dim=0)
        dialogue_kvs = map_all_kvs(lambda x: pad_sequence(torch.repeat_interleave(x, num_generations, dim=0), max_length, 0.0, device, 2), dialogue_kvs)
        log_probs = torch.full((dialogue_lens.shape[0],), 0.0).to(device)
        kls = torch.full((dialogue_lens.shape[0],), math.log(num_generations)-((num_generations-1)/num_generations)).to(device)
        termination_mask = torch.full((dialogue_lens.shape[0],), 1).to(device)
        state_idxs_temp, action_idxs_temp = torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device), torch.zeros((dialogue_lens.shape[0], 1,)).long().to(device)
        t = torch.min(dialogue_lens).int()
        while termination_mask.sum() > 0 and (t+prefix_t) < max_length:
            curr_token = tokens[:, t-1].unsqueeze(1)
            curr_dialogue_kvs = map_all_kvs(lambda x: x[:,:,:(t+prefix_t)-1,:], dialogue_kvs)
            iql_outputs = self.iql_model(curr_token, None, state_idxs_temp, action_idxs_temp, 
                                         policy_kwargs={'use_cache': True, 'past_key_values': curr_dialogue_kvs})
            transformer_outputs, logits = iql_outputs['model_outputs']['policy_model_outputs'], iql_outputs['logits']
            logits[:, 0, tokenizer.pad_token_id] = torch.where(termination_mask == 1, float('-inf'), 1e7)
            logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]] = logits[torch.arange(0, n).to(device), torch.full((n,), 0).to(device), tokens[:, t]].masked_fill_(t < dialogue_lens, 1e7)
            logits = process_logits(logits, temp=temp, top_k=top_k, top_p=top_p)
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
        
        self.iql_model.lm_target = None
        self.iql_model.lm_policy = None
        self.iql_model.model = temp_target

        attn_mask = (tokens != tokenizer.pad_token_id).long()
        if prefix_embs is not None:
            prefix_embs = torch.repeat_interleave(prefix_embs, num_generations, dim=0)
        if prefix_attn_mask is not None:
            prefix_attn_mask = torch.repeat_interleave(prefix_attn_mask, num_generations, dim=0)
        action_idxs = (attn_mask.sum(dim=1)-1).unsqueeze(1)
        state_idxs = (attn_mask.sum(dim=1)-1).unsqueeze(1)
        q_outputs = self.iql_model(tokens, attn_mask, 
                                   state_idxs, action_idxs, 
                                   prefix_embs=prefix_embs, 
                                   prefix_attn_mask=prefix_attn_mask, 
                                   remove_prefix_position_embs=remove_prefix_position_embs)['target_qs']
        q_outputs = q_outputs.squeeze(2).squeeze(1)
        
        self.iql_model.lm_target = temp_target
        self.iql_model.lm_policy = temp_policy
        self.iql_model.model = temp_model

        scores = (q_outputs + log_probs * log_prob_weight).reshape(-1, num_generations)
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
        log_probs = torch.gather(log_probs.reshape(-1, num_generations), dim=1, index=order)
        kls = torch.gather(kls.reshape(-1, num_generations), dim=1, index=order)
        return list(zip(input_strs, processed_outputs)), (log_probs, scores,), kls

    def generate(self, items: InputType, 
                 termination_condition: Callable[[np.ndarray], bool], **kwargs):
        prepared_inputs = self.iql_model.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        state_idxs, action_idxs = prepared_inputs['state_idxs'], prepared_inputs['action_idxs']
        if self.kind == 'rerank':
            method = self.rerank_raw
        else:
            raise NotImplementedError
        generations, info, kls = method(tokens, attn_mask, 
                                             state_idxs, action_idxs, 
                                             termination_condition, 
                                             **kwargs)
        return generations, info, kls
    
    def act(self, obs: Language_Observation) -> str:
        item = DataPoint.from_obs(obs, self.iql_model.dataset.tokenizer, self.iql_model.dataset.token_reward)
        curr_kwargs = dict(self.generation_kwargs)
        best_score = float('-inf')
        selected_items = None
        for _ in range(curr_kwargs.pop('generation_batches')):
            generations, (logprobs, scores,), kls = self.generate([item], always_terminate, **curr_kwargs)
            if scores[0, 0].item() > best_score:
                best_score = scores[0, 0].item()
                selected_items = (generations, (logprobs, scores,), kls)
        (generations, (logprobs, scores,), kls) = selected_items
        self.kls_all.append(kls[0, 0].item())
        self.logprobs_all.append(logprobs[0, 0].item())
        return generations[0][1][0]

class UtteranceIQL_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        policy = PerUtteranceIQL_Policy(model, self.kind, **self.generation_kwargs)
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
