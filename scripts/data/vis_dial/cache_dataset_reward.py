from utils.misc import convert_path
import hydra
from omegaconf import DictConfig, OmegaConf
import visdial.load_objects
from load_objects import load_item
from visdial.visdial_base import QuestionEvent
from visdial.visdial_env import VDObservation, VDRemoteReward
import json
import os
from tqdm.auto import tqdm

def cache_reward(config):
    print(config)
    device = 'cpu'
    dataset = load_item(config['data'], device, verbose=True)
    reward_f = VDRemoteReward(config['reward_url'])
    all_rewards = []
    for i in tqdm(range(dataset.size())):
        item = dataset.get_item(i)
        scene = item.meta['scene']
        rewards = [reward_f.reward(VDObservation(scene, None))]
        for ev in scene.events:
            if isinstance(ev, QuestionEvent):
                continue
            rewards.append(reward_f.reward(VDObservation(scene, ev)))
        all_rewards.append(rewards)
    if not os.path.exists(os.path.dirname(convert_path(config['output_file']))):
        os.makedirs(os.path.dirname(convert_path(config['output_file'])))
    with open(convert_path(config['output_file']), 'w') as f:
        json.dump(all_rewards, f)

@hydra.main(config_path="../../../config/vis_dial", config_name="cache_dataset_reward")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    cache_reward(cfg)

if __name__ == "__main__":
    main()
