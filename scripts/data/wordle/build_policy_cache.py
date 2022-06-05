import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from data.language_environment import interact_environment
from load_objects import load_item
from wordle.wordle_env import WordleEnvironment
from wordle.policy import MixturePolicy

def build_cache(config):
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_policy = load_item(config['cache_policy'], device)
    random_policy = load_item(config['random_policy'], device)
    policy = MixturePolicy(config['cache_policy_mix_prob'], cache_policy, random_policy)
    vocab = load_item(config['vocab'])
    total_reward = 0
    count = 0
    while True:
        result, sequence = interact_environment(WordleEnvironment(vocab), policy)
        reward = sum(map(lambda x: x[2], sequence))
        total_reward += reward
        count += 1
        if count % config['print_every'] == 0:
            print(result)
            print('count:', count)
            print('avg reward:', total_reward / count)
            print('cache hit rate:', cache_policy.cache.get_hit_rate())
            print()
        if count % config['dump_cache_every'] == 0:
            if config['policy_cache_save_path'] is not None:
                cache_policy.cache.dump(config['policy_cache_save_path'])
            if config['vocab_cache_save_path'] is not None:
                vocab.cache.dump(config['vocab_cache_save_path'])


@hydra.main(config_path="../../../config/wordle", config_name="build_policy_cache")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    build_cache(cfg)

if __name__ == "__main__":
    main()