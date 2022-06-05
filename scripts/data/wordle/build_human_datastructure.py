from collections import defaultdict
import csv
import json

from tqdm.auto import tqdm
from wordle.wordle_game import WordleGame, Vocabulary, WordleState
import os
import argparse

char_set = {'🟩': '<g>', '🟨': '<y>', '⬛': '<b>', '⬜': '<b>', '🟧': '<g>', '🟦': '<y>'}
correct_words_map = {'210': 'panic', '211': 'solar', '212': 'shire', '213': 'proxy', 
                     '214': 'point', '215': 'robot', '216': 'prick', 
                     '217': 'wince', '218': 'crimp', '219': 'knoll', '220': 'sugar', 
                     '221': 'whack', '222': 'mount', '223': 'perky', '224': 'could', 
                     '225': 'wrung', '226': 'light', '227': 'those', '228': 'moist', 
                     '229': 'shard', '230': 'pleat', '231': 'aloft', '232': 'skill', 
                     '233': 'elder', '234': 'frame', '235': 'humor', '236': 'pause', 
                     '237': 'ulcer', '238': 'ultra', '239': 'robin', '240': 'cynic', 
                     '241': 'aroma', '242': 'caulk', '243': 'shake', '244': 'dodge', 
                     '245': 'swill', '246': 'tacit', '247': 'other', '248': 'thorn', 
                     '249': 'trove'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--guess_vocab', type=str)
    parser.add_argument('--correct_vocab', type=str)
    parser.add_argument('--tweets_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()

    correct_vocab = Vocabulary.from_file(args.correct_vocab)
    guess_vocab = Vocabulary.from_file(args.guess_vocab)
    transitions = defaultdict(lambda: defaultdict(list))
    for word in tqdm(correct_vocab.all_vocab):
        for act in guess_vocab.all_vocab:
            state = WordleState.initial_state().transition_state(act, word)
            transitions[word][WordleGame(state, guess_vocab, [act]).__repr__().split('</a>')[-1][:-len('</s>')]].append(act)

    games = []
    with open(args.tweets_file, 'r') as f:
        d = csv.reader(f)
        for id, _, _, _, t in d:
            all_rows = t.split('\n')
            game = None
            for i in range(len(all_rows)):
                for x in range(1, 7):
                    if all([len(row) == 5 for row in all_rows[i:(i+x)]]) and all([item in char_set for item in ''.join(all_rows[i:(i+x)])]):
                        game = [''.join([char_set[c] for c in row]) for row in all_rows[i:(i+x)]]
                if game is not None:
                    break
            if game is not None:
                games.append((correct_words_map[id], game))
    
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    with open(args.output_file, 'w') as f:
        json.dump({'games': games, 'transitions': transitions}, f)
