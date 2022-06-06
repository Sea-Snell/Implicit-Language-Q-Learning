from typing import Any, Callable, Dict, Optional, Tuple, List
from data.language_environment import Language_Environment, Language_Observation, Policy
from data.rl_data import List_RL_Dataset, Iterable_RL_Dataset, RL_Dataset
from toxicity.reddit_comments_base import RedditData
import random

class ToxicityObservation(Language_Observation):
    def __init__(self, parent: Optional[str], text: Optional[str], reward: Optional[float]):
        assert (text is None and reward is None) or (text is not None and reward is not None)
        self.parent = parent
        self.text = text
        self.reward = reward
    
    def to_sequence(self) -> Tuple[List[Tuple[str, Optional[float]]], bool]:
        if self.text is None:
            if self.parent is not None:
                return [(self.parent, None)], False
            return [], False
        if self.parent is None:
            return [(self.text, self.reward)], True
        return [(self.parent, None), (self.text, self.reward)], True
    
    def __str__(self) -> str:
        if self.parent is not None:
            return f'parent: {self.parent}\ncomment: {self.text}'
        return self.text

class ToxicityEnvironment(Language_Environment):
    def __init__(self, data: RedditData, 
                 reward_f: Optional[Callable[[str], float]], 
                 reward_shift: float=0.0, reward_scale: float=1.0, 
                 include_parent: bool=True):
        self.data = data
        self.reward_f = reward_f
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale
        self.include_parent = include_parent
        self.stepped = False
        self.parent = None
        self.reset()

    def step(self, action: str) -> Tuple[ToxicityObservation, float, bool]:
        if self.stepped:
            raise Exception("Cannot step after final action")
        self.stepped = True
        reward = (self.reward_f(action) if self.reward_f is not None else 0.0) * self.reward_scale + self.reward_shift
        return ToxicityObservation(self.parent, action, reward), reward, True

    def reset(self) -> ToxicityObservation:
        self.stepped = False
        self.parent = None
        if self.include_parent:
            self.parent = random.choice(self.data)[0][0]
        return ToxicityObservation(self.parent, None, None)

    def is_terminal(self) -> bool:
        return self.stepped

