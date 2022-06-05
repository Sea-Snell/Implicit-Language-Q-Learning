import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objects import load_item
from tqdm.auto import tqdm
from utils.misc import convert_path
import pickle as pkl
from wordle.wordle_env import WordleEnvironment
from data.language_environment import interact_environment

def gen_data(config):
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expert_policy = load_item(config['expert_policy'], device)
    suboptimal_policy = load_item(config['suboptimal_policy'], device)
    vocab = load_item(config['vocab'])
    config['save_path'] = convert_path(config['save_path'])
    if config['load_data'] is not None:
        config['load_data'] = convert_path(config['load_data'])
    raw_cache_save_path = config['vocab']['cache_path']
    if config['expert_policy_cache_save_path'] is not None:
        config['expert_policy_cache_save_path'] = convert_path(config['expert_policy_cache_save_path'])
    if config['suboptimal_policy_cache_save_path'] is not None:
        config['suboptimal_policy_cache_save_path'] = convert_path(config['suboptimal_policy_cache_save_path'])
    if config['vocab_cache_save_path'] is not None:
        raw_cache_save_path = config['vocab_cache_save_path']
        config['vocab_cache_save_path'] = convert_path(config['vocab_cache_save_path'])
    
    total_expert_reward = 0
    expert_count = 0
    total_suboptimal_reward = 0
    suboptimal_count = 0
    all_data = []
    if config['load_data'] is not None:
        with open(config['load_data'], 'rb') as f:
            all_data = pkl.load(f)['state_actions']
    for i in tqdm(range(config['n_trajectories'])):
        expert_wordle_obs, expert_sequence = interact_environment(WordleEnvironment(vocab), expert_policy)
        for x, (o, _, _, _) in enumerate(expert_sequence):
            if o.game.is_terminal():
                break
            for _ in range(config['n_suboptimal']):
                suboptimal_wordle_obs, suboptimal_sequence = interact_environment(WordleEnvironment(vocab), suboptimal_policy, o)
                all_data.append({'state': suboptimal_wordle_obs.game.state, 'actions': suboptimal_wordle_obs.game.action_history, 'meta': {'kind': 'branch_suboptimal', 'start': (o.game.state, o.game.action_history), 'self_actions': suboptimal_wordle_obs.game.action_history}})
                suboptimal_reward = sum(map(lambda x: x[2], expert_sequence[:x]+suboptimal_sequence))
                total_suboptimal_reward += suboptimal_reward
                suboptimal_count += 1
        expert_reward = sum(map(lambda x: x[2], expert_sequence))
        total_expert_reward += expert_reward
        expert_count += 1
        if (config['reward_every'] is not None) and ((i+1) % config['reward_every'] == 0):
            print('avg expert reward:', total_expert_reward / expert_count)
            print('avg suboptimal reward:', total_suboptimal_reward / suboptimal_count)
            print('num data points:', len(all_data))
        all_data.append({'state': expert_wordle_obs.game.state, 'actions': expert_wordle_obs.game.action_history, 'meta': {'kind': 'expert', 'prefixes': [(o.game.state, o.game.action_history) for o, _, _, _ in expert_sequence if not o.game.is_terminal()], 'self_actions': expert_wordle_obs.game.action_history}})
    
    if not os.path.exists(os.path.dirname(config['save_path'])):
        os.makedirs(os.path.dirname(config['save_path']))
    with open(config['save_path'], 'wb') as f:
        pkl.dump({'state_actions': all_data, 
                  'vocab_path': config['vocab']['vocab_path'], 
                  'vocab_cache_path': raw_cache_save_path}, f)
    if config['expert_policy_cache_save_path'] is not None:
        expert_policy.cache.dump(config['expert_policy_cache_save_path'])
    if config['suboptimal_policy_cache_save_path'] is not None:
        suboptimal_policy.cache.dump(config['suboptimal_policy_cache_save_path'])
    if config['vocab_cache_save_path'] is not None:
        vocab.cache.dump(config['vocab_cache_save_path'])

@hydra.main(config_path="../../../config/wordle", config_name="generate_branching_data")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    gen_data(cfg)

if __name__ == "__main__":
    main()
