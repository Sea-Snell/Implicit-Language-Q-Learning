from visdial.visdial_base import PercentileCutoffRule, VisDialogueData
from visdial.visdial_dataset import VisDialListDataset
from visdial.visdial_env import VDEnvironment, VDRemotePolicy
from load_objects import *
import pickle as pkl
from visdial.visdial_evaluator import TopAdvantageUtterances, VisDial_Chai_Evaluator, VisDial_DT_Evaluator, VisDial_IQL_Evaluator, Utterance_VisDial_IQL_Evaluator
from visdial.visdial_tokenizer import gpt3_convert_str_vis_dial, gpt3_convert_token_vis_dial

@register('percentile_cutoff_rule')
def load_percentile_cutoff_rule(config, verbose=True):
    return PercentileCutoffRule(config['goal_value'], 
                                config['percentile'])

@register('vis_dial')
def load_vis_dial(config, verbose=True):
    if config['additional_scenes'] is not None:
        with open(convert_path(config['additional_scenes']), 'rb') as f:
            config['additional_scenes'] = pkl.load(f)
    if config['cutoff_rule'] is not None:
        config['cutoff_rule'] = load_item(config['cutoff_rule'], verbose=verbose)
    return VisDialogueData(convert_path(config['data_path']), 
                           convert_path(config['img_feat_path']), 
                           config['split'], 
                           reward_cache=convert_path(config['reward_cache']), 
                           norm_img_feats=config['norm_img_feats'], 
                           reward_shift=config['reward_shift'], 
                           reward_scale=config['reward_scale'], 
                           addition_scenes=config['additional_scenes'], 
                           mode=config['mode'], 
                           cutoff_rule=config['cutoff_rule'], 
                           yn_reward=config['yn_reward'], 
                           yn_reward_kind=config['yn_reward_kind'])

@register('vis_dial_list_dataset')
def load_vis_list_dataset(config, device, verbose=True):
    vd = load_item(config['data'], verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return VisDialListDataset(vd, max_len=config['max_len'], 
                              token_reward=token_reward, 
                              top_p=config['top_p'], 
                              bottom_p=config['bottom_p'])

@register('vis_dial_env')
def load_vis_env(config, device, verbose=True):
    dataset = load_item(config['dataset'], device, verbose=verbose)
    return VDEnvironment(dataset, config['url'], 
                         reward_shift=config['reward_shift'], 
                         reward_scale=config['reward_scale'], 
                         actor_stop=config['actor_stop'], 
                         yn_reward=config['yn_reward'], 
                         yn_reward_kind=config['yn_reward_kind'])

@register('vis_dial_remote_policy')
def load_vis_dial_remote_policy(config, device, verbose=True):
    return VDRemotePolicy(config['url'])

@register('gpt3_convert_str_vis_dial')
def load_gpt3_convert_str_vis_dial(config, device, verbose=True):
    return gpt3_convert_str_vis_dial

@register('gpt3_convert_token_vis_dial')
def load_gpt3_convert_token_vis_dial(config, device, verbose=True):
    return gpt3_convert_token_vis_dial

@register('top_advantage_utterances_evaluator')
def load_top_advantage_utterances_evaluator(config, device, verbose=True):
    data = load_item(config['data'], device, verbose=verbose)
    return TopAdvantageUtterances(data)

@register('vd_iql_evaluator')
def load_vd_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return VisDial_IQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('vd_dt_evaluator')
def load_vd_dt_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return VisDial_DT_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('utterance_vd_iql_evaluator')
def load_utterance_vd_iql_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return Utterance_VisDial_IQL_Evaluator(env, config['verbose'], config['kind'], **config['generation_kwargs'])

@register('vd_chai_evaluator')
def load_vd_chai_evaluator(config, device, verbose=True):
    env = load_item(config['env'], device, verbose=verbose)
    return VisDial_Chai_Evaluator(env, config['verbose'], config['cache_save_path'], **config['generation_kwargs'])
