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
    adversarial_policy = load_item(config['adversarial_policy'], device)
    vocab = load_item(config['vocab'])
    config['save_path'] = convert_path(config['save_path'])
    if config['load_data'] is not None:
        config['load_data'] = convert_path(config['load_data'])
    raw_cache_save_path = config['vocab']['cache_path']
    if config['expert_policy_cache_save_path'] is not None:
        config['expert_policy_cache_save_path'] = convert_path(config['expert_policy_cache_save_path'])
    if config['adversarial_policy_cache_save_path'] is not None:
        config['adversarial_policy_cache_save_path'] = convert_path(config['adversarial_policy_cache_save_path'])
    if config['suboptimal_policy_cache_save_path'] is not None:
        config['suboptimal_policy_cache_save_path'] = convert_path(config['suboptimal_policy_cache_save_path'])
    if config['vocab_cache_save_path'] is not None:
        raw_cache_save_path = config['vocab_cache_save_path']
        config['vocab_cache_save_path'] = convert_path(config['vocab_cache_save_path'])
    
    total_expert_reward = 0
    expert_count = 0
    total_suboptimal_reward = 0
    suboptimal_count = 0
    total_adversarial_reward = 0
    adversarial_count = 0
    all_data = []
    if config['load_data'] is not None:
        with open(config['load_data'], 'rb') as f:
            all_data = pkl.load(f)['state_actions']
    for i in tqdm(range(config['n_trajectories'])):
        expert_wordle_obs, expert_sequence = interact_environment(WordleEnvironment(vocab), expert_policy)
        expert_reward = sum(map(lambda x: x[2], expert_sequence))
        total_expert_reward += expert_reward
        expert_count += 1
        if len(expert_sequence) > 3:
            all_data.append({'state': expert_wordle_obs.game.state, 'actions': expert_wordle_obs.game.action_history, 'meta': {'kind': 'expert', 's_0': (expert_sequence[0][0].game.state, expert_sequence[0][0].game.action_history), 'a_0': expert_sequence[0][1], 's_2': (expert_sequence[2][0].game.state, expert_sequence[2][0].game.action_history), 'a_2': expert_sequence[2][1]}})
            for _ in range(config['n_adversarial']):
                o = expert_sequence[2][0]
                adversarial_wordle_obs, adversarial_sequence = interact_environment(WordleEnvironment(vocab), adversarial_policy, o)
                adversarial_reward = sum(map(lambda x: x[2], expert_sequence[:2]+adversarial_sequence))
                total_adversarial_reward += adversarial_reward
                adversarial_count += 1
                all_data.append({'state': adversarial_wordle_obs.game.state, 'actions': adversarial_wordle_obs.game.action_history, 'meta': {'kind': 'adversarial', 's_2': (adversarial_sequence[0][0].game.state, adversarial_sequence[0][0].game.action_history), 'a_2': adversarial_sequence[0][1]}})
        else:
            all_data.append({'state': expert_wordle_obs.game.state, 'actions': expert_wordle_obs.game.action_history, 'meta': {'kind': 'expert', 's_0': (expert_sequence[0][0].game.state, expert_sequence[0][0].game.action_history), 'a_0': expert_sequence[0][1]}})
        for _ in range(config['n_suboptimal']):
            suboptimal_wordle_obs, suboptimal_sequence = interact_environment(WordleEnvironment(vocab), suboptimal_policy)
            suboptimal_reward = sum(map(lambda x: x[2], suboptimal_sequence))
            total_suboptimal_reward += suboptimal_reward
            suboptimal_count += 1
            all_data.append({'state': suboptimal_wordle_obs.game.state, 'actions': suboptimal_wordle_obs.game.action_history, 'meta': {'kind': 'suboptimal', 's_0': (suboptimal_sequence[0][0].game.state, suboptimal_sequence[0][0].game.action_history), 'a_0': suboptimal_sequence[0][1]}})
        if (config['reward_every'] is not None) and ((i+1) % config['reward_every'] == 0):
            print('avg expert reward:', total_expert_reward / expert_count)
            print('avg adversarial reward:', total_adversarial_reward / adversarial_count)
            print('avg suboptimal reward:', total_suboptimal_reward / suboptimal_count)
            print('num data points:', len(all_data))
    
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
    if config['adversarial_policy_cache_save_path'] is not None:
        adversarial_policy.cache.dump(config['adversarial_policy_cache_save_path'])
    if config['vocab_cache_save_path'] is not None:
        vocab.cache.dump(config['vocab_cache_save_path'])

@hydra.main(config_path="../../../config/wordle", config_name="generate_adversarial_data")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    gen_data(cfg)

if __name__ == "__main__":
    main()
