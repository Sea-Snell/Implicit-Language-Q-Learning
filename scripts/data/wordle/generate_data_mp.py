import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objects import load_item
from tqdm.auto import tqdm
from utils.misc import convert_path
import pickle as pkl
import multiprocessing as mp
from utils.mp_cache import Cache
from wordle.wordle_env import WordleEnvironment
from data.language_environment import interact_environment

class Worker:
    def __init__(self, q, policy, vocab, dump_every) -> None:
        self.q = q
        self.policy = policy
        self.vocab = vocab
        self.dump_every = dump_every
        self.past_vocab_keys = set(self.vocab.cache.keys())
        self.past_policy_keys = set(self.policy.cache.keys())
    
    def process(self):
        wordle_obs, sequence = interact_environment(WordleEnvironment(self.vocab), self.policy)
        reward = sum(map(lambda x: x[2], sequence))
        self.q.put(({'state': wordle_obs.game.state, 'actions': wordle_obs.game.action_history}, reward,))
    
    def dump_policy(self, mp_cache):
        to_update = {}
        for k, v in tqdm(self.policy.cache.items()):
            if k not in self.past_policy_keys:
                to_update[k] = v
                if len(to_update) % self.dump_every == 0:
                    mp_cache.update(to_update)
                    to_update = {}
        mp_cache.update(to_update)
    
    def dump_vocab(self, mp_cache):
        to_update = {}
        for k, v in tqdm(self.vocab.cache.items()):
            if k not in self.past_vocab_keys:
                to_update[k] = v
                if len(to_update) % self.dump_every == 0:
                    mp_cache.update(to_update)
                    to_update = {}
        mp_cache.update(to_update)

def listener(config, q):
    '''listens for messages on the q, writes to file. '''
    config['save_path'] = convert_path(config['save_path'])
    raw_cache_save_path = config['vocab']['cache_path']
    if config['vocab_cache_save_path'] is not None:
        raw_cache_save_path = config['vocab_cache_save_path']
    if config['load_data'] is not None:
        config['load_data'] = convert_path(config['load_data'])
    total_reward = 0
    all_data = []
    if config['load_data'] is not None:
        with open(config['load_data'], 'rb') as f:
            d = pkl.load(f)
            all_data = d['state_actions']
    while True:
        m = q.get()
        if m == 'kill':
            break
        item, r = m
        all_data.append(item)
        total_reward += r
        if (config['reward_every'] is not None) and (len(all_data) % config['reward_every'] == 0):
            print('avg reward:', total_reward / len(all_data))
    if not os.path.exists(os.path.dirname(config['save_path'])):
        os.makedirs(os.path.dirname(config['save_path']))
    with open(config['save_path'], 'wb') as f:
        pkl.dump({'state_actions': all_data, 
                  'vocab_path': config['vocab']['vocab_path'], 
                  'vocab_cache_path': raw_cache_save_path}, f)
    return None

def init(q, policy, vocab, dump_every):
    global worker
    worker = Worker(q, policy, vocab, dump_every)

def process(_):
    global worker
    worker.process()

def dump_policy(_):
    global worker
    global mp_policy_cache
    worker.dump_policy(mp_policy_cache)

def dump_vocab(_):
    global worker
    global mp_vocab_cache
    worker.dump_vocab(mp_vocab_cache)

def gen_data(config):
    print(config)
    global worker
    global mp_vocab_cache
    global mp_policy_cache
    worker = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = load_item(config['policy'], device)
    vocab = load_item(config['vocab'])
    mp_vocab_cache = Cache(vocab.cache.get_cache())
    mp_policy_cache = Cache(policy.cache.get_cache())

    print('setting up...')
    q = mp.Manager().Queue()
    p = mp.Process(target=listener, args=(config, q,))
    p.start()
    
    with mp.Pool(config['n_processes'], initializer=init, initargs=(q, policy, vocab, config['dump_every'])) as pool:
        print('starting...')
        _ = list(tqdm(pool.imap(process, range(config['n_trajectories'])), total=config['n_trajectories']))
        print('combining policy cache...')
        if config['policy_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_policy, range(config['n_processes'])), total=config['n_processes']))
        print('combining vocab cache...')
        if config['vocab_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_vocab, range(config['n_processes'])), total=config['n_processes']))
        print('saving trajectories...')
        q.put('kill')
        p.join()
    print('saving policy cache...')
    if config['policy_cache_save_path'] is not None:
        mp_policy_cache.dump(convert_path(config['policy_cache_save_path']))
    print('saving vocab cache...')
    if config['vocab_cache_save_path'] is not None:
        mp_vocab_cache.dump(convert_path(config['vocab_cache_save_path']))

@hydra.main(config_path="../../../config/wordle", config_name="generate_data_mp")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    gen_data(cfg)

if __name__ == "__main__":
    main()
