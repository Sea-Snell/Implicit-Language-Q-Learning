import os
import csv
from typing import Callable, List, Optional
from utils.cache import Cache

class RedditData:
    def __init__(self, path: str, 
                 indexes: Optional[List[int]], 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_cache: Optional[Cache]=None, 
                 reward_shift: float=0.0, 
                 reward_scale: float=1.0):
        with open(os.path.join(path, 'comments_positive.csv'), 'r') as f:
            items = [row for row in csv.reader(f)][1:]
        with open(os.path.join(path, 'comments_negative.csv'), 'r') as f:
            items += [row for row in csv.reader(f)][1:]
        if indexes is not None:
            items = [items[idx] for idx in indexes]
        self.info = (path, len(indexes))
        self.reward_cache = reward_cache
        if self.reward_cache is None:
            self.reward_cache = Cache()
        self.data = [item[4] for item in items]
        self.parent_data = [item[10] for item in items]
        self.gt_scores = [int(item[5]) for item in items]
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.reward_f = reward_f

    def __getitem__(self, idx):
        comment = self.data[idx]
        parent = self.parent_data[idx]
        if comment not in self.reward_cache:
            self.reward_cache[comment] = self.reward_f(comment) if self.reward_f is not None else 0.0
        return (parent, comment,), self.reward_cache[comment] * self.reward_scale + self.reward_shift
    
    def __len__(self):
        return len(self.data)

