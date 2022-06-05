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
    def __init__(self, q, expert_policy, suboptimal_policy, adversarial_policy, vocab, dump_every, n_suboptimal, n_adversarial) -> None:
        self.q = q
        self.expert_policy = expert_policy
        self.suboptimal_policy = suboptimal_policy
        self.adversarial_policy = adversarial_policy
        self.vocab = vocab
        self.dump_every = dump_every
        self.past_vocab_keys = set(self.vocab.cache.keys())
        self.past_expert_policy_keys = set(self.expert_policy.cache.keys())
        self.past_suboptimal_policy_keys = set(self.suboptimal_policy.cache.keys())
        self.past_adversarial_policy_keys = set(self.adversarial_policy.cache.keys())
        self.n_suboptimal = n_suboptimal
        self.n_adversarial = n_adversarial
    
    def process(self):
        expert_wordle_obs, expert_sequence = interact_environment(WordleEnvironment(self.vocab), self.expert_policy)
        expert_reward = sum(map(lambda x: x[2], expert_sequence))
        if len(expert_sequence) > 3:
            self.q.put(({'state': expert_wordle_obs.game.state, 'actions': expert_wordle_obs.game.action_history, 'meta': {'kind': 'expert', 's_0': (expert_sequence[0][0].game.state, expert_sequence[0][0].game.action_history), 'a_0': expert_sequence[0][1], 's_2': (expert_sequence[2][0].game.state, expert_sequence[2][0].game.action_history), 'a_2': expert_sequence[2][1]}}, expert_reward,))
            for _ in range(self.n_adversarial):
                o = expert_sequence[2][0]
                adversarial_wordle_obs, adversarial_sequence = interact_environment(WordleEnvironment(self.vocab), self.adversarial_policy, o)
                adversarial_reward = sum(map(lambda x: x[2], expert_sequence[:2]+adversarial_sequence))
                self.q.put(({'state': adversarial_wordle_obs.game.state, 'actions': adversarial_wordle_obs.game.action_history, 'meta': {'kind': 'adversarial', 's_2': (adversarial_sequence[0][0].game.state, adversarial_sequence[0][0].game.action_history), 'a_2': adversarial_sequence[0][1]}}, adversarial_reward,))
        else:
            self.q.put(({'state': expert_wordle_obs.game.state, 'actions': expert_wordle_obs.game.action_history, 'meta': {'kind': 'expert', 's_0': (expert_sequence[0][0].game.state, expert_sequence[0][0].game.action_history), 'a_0': expert_sequence[0][1]}}, expert_reward,))
        for _ in range(self.n_suboptimal):
            suboptimal_wordle_obs, suboptimal_sequence = interact_environment(WordleEnvironment(self.vocab), self.suboptimal_policy)
            suboptimal_reward = sum(map(lambda x: x[2], suboptimal_sequence))
            self.q.put(({'state': suboptimal_wordle_obs.game.state, 'actions': suboptimal_wordle_obs.game.action_history, 'meta': {'kind': 'suboptimal', 's_0': (suboptimal_sequence[0][0].game.state, suboptimal_sequence[0][0].game.action_history), 'a_0': suboptimal_sequence[0][1]}}, suboptimal_reward))

    def dump_expert_policy(self, mp_cache):
        to_update = {}
        for k, v in tqdm(self.expert_policy.cache.items()):
            if k not in self.past_expert_policy_keys:
                to_update[k] = v
                if len(to_update) % self.dump_every == 0:
                    mp_cache.update(to_update)
                    to_update = {}
        mp_cache.update(to_update)
    
    def dump_suboptimal_policy(self, mp_cache):
        to_update = {}
        for k, v in tqdm(self.suboptimal_policy.cache.items()):
            if k not in self.past_suboptimal_policy_keys:
                to_update[k] = v
                if len(to_update) % self.dump_every == 0:
                    mp_cache.update(to_update)
                    to_update = {}
        mp_cache.update(to_update)
    
    def dump_adversarial_policy(self, mp_cache):
        to_update = {}
        for k, v in tqdm(self.adversarial_policy.cache.items()):
            if k not in self.past_adversarial_policy_keys:
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
    total_expert_reward = 0
    total_suboptimal_reward = 0
    total_adversarial_reward = 0
    expert_count = 0
    suboptimal_count = 0
    adversarial_count = 0
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
        if item['meta']['kind'] == 'expert':
            total_expert_reward += r
            expert_count += 1
        elif item['meta']['kind'] == 'suboptimal':
            total_suboptimal_reward += r
            suboptimal_count += 1
        elif item['meta']['kind'] == 'adversarial':
            total_adversarial_reward += r
            adversarial_count += 1
        else:
            raise NotImplementedError
        if (config['reward_every'] is not None) and (len(all_data) % config['reward_every'] == 0):
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
    return None

def init(q, expert_policy, suboptimal_policy, adversarial_policy, vocab, dump_every, n_suboptimal, n_adversarial):
    global worker
    worker = Worker(q, expert_policy, suboptimal_policy, adversarial_policy, vocab, dump_every, n_suboptimal, n_adversarial)

def process(_):
    global worker
    worker.process()

def dump_expert_policy(_):
    global worker
    global mp_expert_policy_cache
    worker.dump_expert_policy(mp_expert_policy_cache)

def dump_suboptimal_policy(_):
    global worker
    global mp_suboptimal_policy_cache
    worker.dump_suboptimal_policy(mp_suboptimal_policy_cache)

def dump_adversarial_policy(_):
    global worker
    global mp_adversarial_policy_cache
    worker.dump_adversarial_policy(mp_adversarial_policy_cache)

def dump_vocab(_):
    global worker
    global mp_vocab_cache
    worker.dump_vocab(mp_vocab_cache)

def gen_data(config):
    print(config)
    global worker
    global mp_vocab_cache
    global mp_expert_policy_cache
    global mp_suboptimal_policy_cache
    global mp_adversarial_policy_cache
    worker = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expert_policy = load_item(config['expert_policy'], device)
    suboptimal_policy = load_item(config['suboptimal_policy'], device)
    adversarial_policy = load_item(config['adversarial_policy'], device)
    vocab = load_item(config['vocab'])
    mp_vocab_cache = Cache(vocab.cache.get_cache())
    print(len(mp_vocab_cache))
    mp_expert_policy_cache = Cache(expert_policy.cache.get_cache())
    mp_suboptimal_policy_cache = Cache(suboptimal_policy.cache.get_cache())
    mp_adversarial_policy_cache = Cache(adversarial_policy.cache.get_cache())

    print('setting up...')
    q = mp.Manager().Queue()
    p = mp.Process(target=listener, args=(config, q,))
    p.start()

    with mp.Pool(config['n_processes'], initializer=init, initargs=(q, expert_policy, suboptimal_policy, adversarial_policy, vocab, config['dump_every'], config['n_suboptimal'], config['n_adversarial'])) as pool:
        print('starting...')
        _ = list(tqdm(pool.imap(process, range(config['n_trajectories'])), total=config['n_trajectories']))
        print('combining expert policy cache...')
        if config['expert_policy_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_expert_policy, range(config['n_processes'])), total=config['n_processes']))
        print('combining suboptimal policy cache...')
        if config['suboptimal_policy_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_suboptimal_policy, range(config['n_processes'])), total=config['n_processes']))
        print('combining adversarial policy cache...')
        if config['adversarial_policy_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_adversarial_policy, range(config['n_processes'])), total=config['n_processes']))
        print('combining vocab cache...')
        if config['vocab_cache_save_path'] is not None:
            _ = list(tqdm(pool.imap(dump_vocab, range(config['n_processes'])), total=config['n_processes']))
        print('saving trajectories...')
        q.put('kill')
        p.join()
    print(len(mp_vocab_cache))
    print('saving expert policy cache...')
    if config['expert_policy_cache_save_path'] is not None:
        mp_expert_policy_cache.dump(convert_path(config['expert_policy_cache_save_path']))
    print('saving suboptimal policy cache...')
    if config['suboptimal_policy_cache_save_path'] is not None:
        mp_suboptimal_policy_cache.dump(convert_path(config['suboptimal_policy_cache_save_path']))
    print('saving adversarial policy cache...')
    if config['adversarial_policy_cache_save_path'] is not None:
        mp_adversarial_policy_cache.dump(convert_path(config['adversarial_policy_cache_save_path']))
    print('saving vocab cache...')
    if config['vocab_cache_save_path'] is not None:
        mp_vocab_cache.dump(convert_path(config['vocab_cache_save_path']))

@hydra.main(config_path="../../../config/wordle", config_name="generate_adversarial_data_mp")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    gen_data(cfg)

if __name__ == "__main__":
    main()
