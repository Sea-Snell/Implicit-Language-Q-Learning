import sys
import os
from data.rl_data import Iterable_RL_Dataset
from data.torch_datasets import GeneralDataset, GeneralIterDataset
from utils.misc import add_system_configs, strip_from_beginning, strip_from_end
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objects import load_item
from accelerate import Accelerator
import toxicity.load_objects
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm.auto import tqdm
# from utils.mp_cache import Cache
from utils.cache import Cache
import torch

def cache_rewards(cfg):
    print('using config:', cfg)
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])

    raw_dataset = load_item(cfg['dataset'], system_cfg['device'])
    if isinstance(raw_dataset, Iterable_RL_Dataset):
        dataset = GeneralIterDataset(raw_dataset, 'cpu')
    else:
        dataset = GeneralDataset(raw_dataset, 'cpu')
    data_loader_kwargs = {'num_workers': cfg['dataloader_workers'], 
                          'batch_size': cfg['bsize'], 
                          'collate_fn': dataset.collate}
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = True
    data_loader = DataLoader(dataset, **data_loader_kwargs)

    model = load_item(cfg['model'], system_cfg['device'])
    model.eval()
    
    if not isinstance(dataset, IterableDataset):
        data_loader = accelerator.prepare(data_loader)

    cache = Cache()
    with torch.no_grad():
        for items in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            comments = [strip_from_end(strip_from_beginning((comment[:comment.find('<|pad|>')] if comment.find('<|pad|>') != -1 else comment).strip(), '<a>'), '</a> </eod>').strip()  for comment in model.dataset.tokenizer.decode(items['tokens'].detach().cpu().tolist())]
            rewards = model.get_reward_raw(items)
            for i in range(cfg['bsize']):
                cache[comments[i]] = rewards[i]

@hydra.main(config_path="../../../config/toxicity", config_name="cache_model_rewards")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    cache_rewards(cfg)

if __name__ == "__main__":
    main()