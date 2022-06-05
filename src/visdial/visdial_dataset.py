from typing import Optional
from data.rl_data import DataPoint, List_RL_Dataset, TokenReward
from visdial.visdial_base import VisDialogueData
from visdial.visdial_env import VDObservation
from visdial.visdial_tokenizer import VisDialTokenizer
import numpy as np

class VisDialListDataset(List_RL_Dataset):
    def __init__(self, data: VisDialogueData, 
                 max_len: Optional[int], 
                 token_reward: TokenReward, 
                 top_p: Optional[float]=None, 
                 bottom_p: Optional[float]=None, 
                ) -> None:
        tokenizer = VisDialTokenizer()
        super().__init__(tokenizer, token_reward, max_len)
        self.data = data
        self.datapoints = []
        for item in self.data:
            obs = VDObservation(item, item.events[-1])
            self.datapoints.append(DataPoint.from_obs(obs, self.tokenizer, self.token_reward))
        if bottom_p is not None:
            total_rs = [sum(item.rewards) for item in self.datapoints]
            top_idxs = np.argsort(total_rs)
            filtered_idxs = top_idxs[:int(len(top_idxs)*bottom_p)]
            self.datapoints = [self.datapoints[idx] for idx in filtered_idxs]
            self.data = [self.data[idx] for idx in filtered_idxs]
        if top_p is not None:
            total_rs = [sum(item.rewards) for item in self.datapoints]
            top_idxs = np.argsort(total_rs)
            filtered_idxs = top_idxs[-int(len(top_idxs)*top_p):]
            self.datapoints = [self.datapoints[idx] for idx in filtered_idxs]
            self.data = [self.data[idx] for idx in filtered_idxs]
    
    def get_item(self, idx: int):
        return self.datapoints[idx]
    
    def size(self):
        return len(self.datapoints)
