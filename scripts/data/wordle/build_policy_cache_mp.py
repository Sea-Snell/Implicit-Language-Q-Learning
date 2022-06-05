import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from data.language_environment import interact_environment
from load_objects import load_item
from wordle.wordle_env import WordleEnvironment
from wordle.policy import MixturePolicy
from utils.mp_cache import Cache
from utils.misc import convert_path
import multiprocessing as mp

def combine_results(config, q, policy_cache, vocab_cache):
    count = 0
    total_reward = 0
    while True:
        reward = q.get()
        total_reward += reward
        count += 1
        if count % config['print_every'] == 0:
            print('count:', count)
            print('avg reward:', total_reward / count)
            print('cache hit rate:', policy_cache.get_hit_rate())
            print()
        if count % config['dump_cache_every'] == 0:
            print('dumping cache ...')
            if config['policy_cache_save_path'] is not None:
                policy_cache.dump(convert_path(config['policy_cache_save_path']))
            if config['vocab_cache_save_path'] is not None:
                vocab_cache.dump(convert_path(config['vocab_cache_save_path']))
            print('dumped.')

        
def process_trajectories(q, policy, vocab):
    while True:
        _, sequence = interact_environment(WordleEnvironment(vocab), policy)
        reward = sum(map(lambda x: x[2], sequence))
        q.put(reward)

def build_cache(config):
    print(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_policy = load_item(config['cache_policy'], device)
    cache_policy.cache = Cache(cache_policy.cache.get_cache())
    random_policy = load_item(config['random_policy'], device)
    policy = MixturePolicy(config['cache_policy_mix_prob'], cache_policy, random_policy)
    vocab = load_item(config['vocab'])
    vocab.cache = Cache(vocab.cache.get_cache())

    q = mp.Manager().Queue()
    print('starting combiner ...')
    p = mp.Process(target=combine_results, args=(config, q, cache_policy.cache, vocab.cache,))
    p.start()
    print('started.')

    print('starting workers ...')
    main_ps = [mp.Process(target=process_trajectories, args=(q, policy, vocab,)) for _ in range(config['n_processes'])]
    for main_p in main_ps:
        main_p.start()
    for main_p in main_ps:
        main_p.join()

@hydra.main(config_path="../../../config/wordle", config_name="build_policy_cache_mp")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    build_cache(cfg)

if __name__ == "__main__":
    mp.set_start_method('fork')
    main()