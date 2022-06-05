from typing import Any, Dict, Optional
from models.base import Evaluator, InputType
from models.dt_model import DT_Policy
from models.iql_model import IQL_Policy, PerTokenIQL
from collections import defaultdict
from data.language_environment import Language_Environment, interact_environment
from data.rl_data import DataPoint
from models.utterance_iql_model import PerUtteranceIQL_Policy
from models.chai_model import ChaiModel, ChaiPolicy
from visdial.visdial_base import AnswerEvent, QuestionEvent
from visdial.visdial_dataset import VisDialListDataset
from tqdm import tqdm
import torch
from collections import defaultdict
import numpy as np
import time

class TopAdvantageUtterances(Evaluator):
    def __init__(self, data: VisDialListDataset) -> None:
        self.data = data
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        top_actions = defaultdict(float)
        total_actions = defaultdict(int)
        for i in tqdm(range(self.data.size())):
            item = self.data.get_item(i)
            prepared_inputs = model.prepare_inputs([item])
            tokens, a_idx = prepared_inputs['tokens'], prepared_inputs['action_idxs']
            model_outputs = model.get_qvs([item])
            select_tokens = torch.gather(tokens[0, 1:], dim=0, index=a_idx[0, :])
            advantages = model_outputs['target_qs'][0, :] - model_outputs['target_vs'][0, :]
            curr_idx = 0
            for x, token in enumerate(select_tokens):
                if select_tokens[x].item() == self.data.tokenizer.eoa_token_id:
                    total_advantage = advantages[curr_idx:x].sum().item()
                    utterance = self.data.tokenizer.decode(tokens[0, (a_idx[0, curr_idx].item()+1):(a_idx[0, x].item()+1)].detach().cpu().tolist())
                    top_actions[utterance] += total_advantage
                    total_actions[utterance] += 1
                    curr_idx = x+1
            if i % 100 == 0:
                ranked_actions = sorted({k: top_actions[k] / total_actions[k] for k in total_actions.keys()}.items(), key=lambda x: x[1])
                print(ranked_actions[-10:])
                print(ranked_actions[:10])


class VisDial_IQL_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.act_counts = []
        self.all_results = []
        # self.all_entropy = []
        self.all_time = []
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        policy = IQL_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        total_token_reward = 0
        total_env_reward = 0
        total_activation_count = 0
        for i in range(tokens.shape[0]):
            s = time.time()
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            activation_count = sum(map(int, [self.env.yn_reward_f(ev.answer) if self.env.yn_reward_f is not None else 0 for ev in result.event.get_events() if isinstance(ev, AnswerEvent)]))
            self.act_counts.append(activation_count / (len(result.event.get_events())/2))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_activation_count += activation_count
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('activation count:', activation_count)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg activation count:', total_activation_count / (i + 1))
                print('='*25)
            e = time.time()
            self.all_time.append(e-s)
        kl_total = sum(policy.kls_all)
        time_total = sum(self.all_time)
        print(np.histogram(self.act_counts))
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 
                'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 
                'kl': (kl_total / len(policy.kls_all), len(policy.kls_all)), 
                'activation_count': (total_activation_count / tokens.shape[0], tokens.shape[0]), 
                'time': (time_total / len(self.all_time), len(self.all_time))}

    def dump(self):
        return {'results': self.all_results, 'histogram': self.act_counts, 'time': self.all_time}

class Utterance_VisDial_IQL_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.act_counts = []
        self.all_results = []
        self.all_entropy = []
        self.all_time = []
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        policy = PerUtteranceIQL_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        total_token_reward = 0
        total_env_reward = 0
        total_activation_count = 0
        for i in range(tokens.shape[0]):
            s = time.time()
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            activation_count = sum(map(int, [self.env.yn_reward_f(ev.answer) if self.env.yn_reward_f is not None else 0 for ev in result.event.get_events() if isinstance(ev, AnswerEvent)]))
            self.act_counts.append(activation_count / (len(result.event.get_events())/2))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_activation_count += activation_count
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('activation count:', activation_count)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg activation count:', total_activation_count / (i + 1))
                print('='*25)
            e = time.time()
            self.all_time.append(e-s)
        kl_total = sum(policy.kls_all)
        entropy_total = -sum(policy.logprobs_all)
        time_total = sum(self.all_time)
        self.all_entropy.extend(policy.logprobs_all)
        print(np.histogram(self.act_counts))
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 
                'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 
                'kl': (kl_total / len(policy.kls_all), len(policy.kls_all)), 
                'activation_count': (total_activation_count / tokens.shape[0], tokens.shape[0]), 
                'entropy': (entropy_total / len(policy.logprobs_all), len(policy.logprobs_all)), 
                'time': (time_total / len(self.all_time), len(self.all_time))}

    def dump(self):
        return {'results': self.all_results, 'histogram': self.act_counts, 'entropies': self.all_entropy, 'time': self.all_time}

class VisDial_DT_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, kind: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.kind = kind
        self.generation_kwargs = generation_kwargs
        self.act_counts = []
        self.all_results = []
    
    def evaluate(self, model: PerTokenIQL, items: InputType) -> Optional[Dict[str, Any]]:
        policy = DT_Policy(model, self.kind, **self.generation_kwargs)
        tokens = model.prepare_inputs(items)['tokens']
        total_token_reward = 0
        total_env_reward = 0
        total_activation_count = 0
        for i in range(tokens.shape[0]):
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            activation_count = sum(map(int, [self.env.yn_reward_f(ev.answer) for ev in result.event.get_events() if isinstance(ev, AnswerEvent)]))
            self.act_counts.append(activation_count / (len(result.event.get_events())/2))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_activation_count += activation_count
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('activation count:', activation_count)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg activation count:', total_activation_count / (i + 1))
                print('='*25)
        print(np.histogram(self.act_counts))
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 
                'activation_count': (total_activation_count / tokens.shape[0], tokens.shape[0])}

    def dump(self):
        return {'results': self.all_results, 'histogram': self.act_counts}

class VisDial_Chai_Evaluator(Evaluator):
    def __init__(self, env: Language_Environment, verbose: bool, cache_save_path: str, **generation_kwargs) -> None:
        super().__init__()
        self.env = env
        self.verbose = verbose
        self.cache_save_path = cache_save_path
        self.generation_kwargs = generation_kwargs
        self.act_counts = []
        self.all_results = []
        self.all_time = []
    
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
        total_activation_count = 0
        for i in range(tokens.shape[0]):
            s = time.time()
            result, sequence = interact_environment(self.env, policy, None)
            self.all_results.append((result, sequence,))
            activation_count = sum(map(int, [self.env.yn_reward_f(ev.answer) for ev in result.event.get_events() if isinstance(ev, AnswerEvent)]))
            self.act_counts.append(activation_count / (len(result.event.get_events())/2))
            env_reward = sum(map(lambda x: x[2], sequence))
            token_reward = sum(DataPoint.get_token_reward(result, model.dataset.tokenizer, model.dataset.token_reward))
            total_env_reward += env_reward
            total_token_reward += token_reward
            total_activation_count += activation_count
            if self.verbose:
                print(result)
                print('='*25)
                print('token reward:', token_reward)
                print('env reward:', env_reward)
                print('activation count:', activation_count)
                print('avg token reward:', total_token_reward / (i + 1))
                print('avg env reward:', total_env_reward / (i + 1))
                print('avg activation count:', total_activation_count / (i + 1))
                print('='*25)
            e = time.time()
            self.all_time.append(e-s)
        time_total = sum(self.all_time)
        print(np.histogram(self.act_counts))
        return {'token_reward': (total_token_reward / tokens.shape[0], tokens.shape[0]), 'env_reward': (total_env_reward / tokens.shape[0], tokens.shape[0]), 
                'activation_count': (total_activation_count / tokens.shape[0], tokens.shape[0]), 
                'time': (time_total / len(self.all_time), len(self.all_time))}
    
    def dump(self):
        return {'results': self.all_results, 'histogram': self.act_counts, 'time': self.all_time}
