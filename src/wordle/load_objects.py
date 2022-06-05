from load_objects import *
from wordle.policy import MixturePolicy, MonteCarloPolicy, OptimalPolicy, RandomMixturePolicy, RepeatPolicy, StartWordPolicy, UserPolicy, WrongPolicy
from wordle.wordle_dataset import WordleHumanDataset, WordleIterableDataset, WordleListDataset
from wordle.wordle_env import WordleEnvironment
from wordle.wordle_game import Vocabulary
from wordle.wordle_evaluators import Action_Ranking_Evaluator, Action_Ranking_Evaluator_Adversarial

@register('vocab')
def load_vocab(config, verbose=True):
    vocab = Vocabulary.from_file(convert_path(config['vocab_path']), config['fill_cache'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading vocab cache from: %s' % convert_path(config['cache_path']))
        vocab.cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return vocab

@register('wordle_env')
def load_wordle_environment(config, device, verbose=True):
    vocab = load_item(config['vocab'], verbose=verbose)
    return WordleEnvironment(vocab)

@register('user_policy')
def load_user_policy(config, device, verbose=True):
    vocab, hint_policy = None, None
    if config['hint_policy'] is not None:
        hint_policy = load_item(config['hint_policy'], device, verbose=verbose)
    if config['vocab'] is not None:
        vocab = load_item(config['vocab'], verbose=verbose)
    return UserPolicy(hint_policy=hint_policy, vocab=vocab)

@register('start_word_policy')
def load_start_word_policy(config, device, verbose=True):
    return StartWordPolicy(config['start_words'])

@register('optimal_policy')
def load_optimal_policy(config, device, verbose=True):
    start_word_policy = None
    if config['start_word_policy'] is not None:
        start_word_policy = load_item(config['start_word_policy'], device, verbose=verbose)
    policy = OptimalPolicy(start_word_policy=start_word_policy, progress_bar=config['progress_bar'])
    if config['cache_path'] is not None:
        if verbose:
            print('loading optimal policy cache from: %s' % convert_path(config['cache_path']))
        policy.cache.load(convert_path(config['cache_path']))
        if verbose:
            print('loaded.')
    return policy

@register('wrong_policy')
def load_wrong_policy(config, device, verbose=True):
    vocab = load_item(config['vocab'], verbose=verbose)
    return WrongPolicy(vocab=vocab)

@register('repeat_policy')
def load_repeat_policy(config, device, verbose=True):
    start_word_policy = None
    if config['start_word_policy'] is not None:
        start_word_policy = load_item(config['start_word_policy'], device, verbose=verbose)
    return RepeatPolicy(start_word_policy=start_word_policy, first_n=config['first_n'])

@register('mixture_policy')
def load_mixture_policy(config, device, verbose=True):
    policy1 = load_item(config['policy1'], device, verbose=verbose)
    policy2 = load_item(config['policy2'], device, verbose=verbose)
    return MixturePolicy(config['prob1'], policy1, policy2)

@register('random_mixture_policy')
def load_random_mixture_policy(config, device, verbose=True):
    vocab = None
    if config['vocab'] is not None:
        vocab = load_item(config['vocab'], verbose=verbose)
    return RandomMixturePolicy(prob_smart=config['prob_smart'], vocab=vocab)

@register('monte_carlo_policy')
def load_monte_carlo_policy(config, device, verbose=True):
    sample_policy = load_item(config['sample_policy'], device, verbose=verbose)
    return MonteCarloPolicy(n_samples=config['n_samples'], sample_policy=sample_policy)

@register('wordle_iterable_dataset')
def load_wordle_iterable_dataset(config, device, verbose=True):
    policy = load_item(config['policy'], device, verbose=verbose)
    vocab = load_item(config['vocab'], verbose=verbose)
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return WordleIterableDataset(policy, vocab, max_len=config['max_len'], token_reward=token_reward)

@register('wordle_dataset')
def load_wordle_dataset(config, device, verbose=True):
    if config['vocab'] is not None:
        vocab = load_item(config['vocab'], verbose=verbose)
    else:
        vocab = None
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    return WordleListDataset.from_file(convert_path(config['file_path']), config['max_len'], vocab, token_reward)

@register('wordle_human_dataset')
def load_human_dataset(config, device, verbose=True):
    token_reward = load_item(config['token_reward'], device, verbose=verbose)
    game_indexes = None
    if config['index_file'] is not None:
        with open(convert_path(config['index_file']), 'r') as f:
            game_indexes = json.load(f)
    return WordleHumanDataset.from_file(convert_path(config['file_path']), config['use_true_word'], config['max_len'], token_reward, 
                                        game_indexes, config['top_p'])

@register('action_ranking_evaluator')
def load_action_ranking_evaluator(config, device, verbose=True):
    branching_data = load_item(config['branching_data'], device, verbose=verbose)
    return Action_Ranking_Evaluator(branching_data)

@register('action_ranking_evaluator_adversarial')
def load_action_ranking_evaluator_adversarial(config, device, verbose=True):
    adversarial_data = load_item(config['adversarial_data'], device, verbose=verbose)
    return Action_Ranking_Evaluator_Adversarial(adversarial_data)

