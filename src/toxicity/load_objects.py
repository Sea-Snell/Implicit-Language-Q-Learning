from load_objects import *
import json
from toxicity.reddit_comments_base import RedditData
from toxicity.reward_fs import model_reward, score_human_reward, toxicity_reward, toxicity_noised_reward
from toxicity.reward_model import RobertaBinaryRewardModel
from toxicity.toxicity_dataset import ToxicityListDataset
from toxicity.toxicity_env import ToxicityEnvironment

@register('toxicity_reward')
def load_toxicity_reward(config, device, verbose=True):
    return toxicity_reward()

@register('toxicity_noised_reward')
def load_toxicity_noised_reward(config, device, verbose=True):
    return toxicity_noised_reward()

@register('score_human_reward')
def load_score_human_reward(config, device, verbose=True):
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    return score_human_reward(convert_path(config['reddit_path']), indexes)

@register('model_reward')
def load_model_reward(config, device, verbose=True):
    model = load_item(config['model'], device, verbose=verbose)
    return model_reward(model)

@register('reddit_comments')
def load_reddit_comments(config, device, verbose=True):
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    if config['index_path'] is not None:
        with open(convert_path(config['index_path']), 'r') as f:
            indexes = json.load(f)
    else:
        indexes = None
    data = RedditData(convert_path(config['path']), indexes, reward_f, None, 
                      config['reward_shift'], config['reward_scale'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading reddit reward cache from: %s' % convert_path(config['cache_path']))
        data.reward_cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return data

@register('toxicity_list_dataset')
def load_toxicity_list_dataset(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return ToxicityListDataset(data, max_len=config['max_len'], 
                               token_reward=token_reward, 
                               cuttoff=config['cuttoff'], 
                               resample_timeout=config['resample_timeout'], 
                               include_parent=config['include_parent'])

@register('toxicity_env')
def load_toxicity_env(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    if config['reward_f'] is not None:
        reward_f = load_item(config['reward_f'], device, verbose=verbose)
    else:
        reward_f = None
    return ToxicityEnvironment(data=data, 
                               reward_f=reward_f, 
                               reward_shift=config['reward_shift'], 
                               reward_scale=config['reward_scale'], 
                               include_parent=config['include_parent'])

@register('roberta_binary_reward_model')
def load_roberta_binary_reward_model(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    model = RobertaBinaryRewardModel(dataset, device, config['roberta_kind'], 
                                     freeze_roberta=config['freeze_roberta'], 
                                     reward_cuttoff=config['reward_cuttoff'])
    return load_model(config['load'], model, device, verbose=verbose)

