import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objects import load_item
from tqdm.auto import tqdm
from utils.misc import convert_path
from wordle.wordle_env import WordleEnvironment
from data.language_environment import interact_environment
import pickle as pkl

def gen_data(config):
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_item(config['policy'], device)
    vocab = load_item(config['vocab'])
    config['save_path'] = convert_path(config['save_path'])
    if config['policy_cache_save_path'] is not None:
        config['policy_cache_save_path'] = convert_path(config['policy_cache_save_path'])
    raw_cache_save_path = config['vocab']['cache_path']
    if config['vocab_cache_save_path'] is not None:
        raw_cache_save_path = config['vocab_cache_save_path']
        config['vocab_cache_save_path'] = convert_path(config['vocab_cache_save_path'])
    if config['load_data'] is not None:
        config['load_data'] = convert_path(config['load_data'])
    
    total_reward = 0
    all_data = []
    if config['load_data'] is not None:
        with open(config['load_data'], 'rb') as f:
            all_data = pkl.load(f)['state_actions']
    for i in tqdm(range(config['n_trajectories'])):
        wordle_obs, sequence = interact_environment(WordleEnvironment(vocab), policy)
        reward = sum(map(lambda x: x[2], sequence))
        total_reward += reward
        if (config['reward_every'] is not None) and ((i+1) % config['reward_every'] == 0):
            print('avg reward:', total_reward / (i+1))
        all_data.append({'state': wordle_obs.game.state, 'actions': wordle_obs.game.action_history})
    
    if not os.path.exists(os.path.dirname(config['save_path'])):
        os.makedirs(os.path.dirname(config['save_path']))
    with open(config['save_path'], 'wb') as f:
        pkl.dump({'state_actions': all_data, 
                  'vocab_path': config['vocab']['vocab_path'], 
                  'vocab_cache_path': raw_cache_save_path}, f)
    if config['policy_cache_save_path'] is not None:
        policy.cache.dump(config['policy_cache_save_path'])
    if config['vocab_cache_save_path'] is not None:
        vocab.cache.dump(config['vocab_cache_save_path'])

@hydra.main(config_path="../../../config/wordle", config_name="generate_data")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    gen_data(cfg)

if __name__ == "__main__":
    main()
