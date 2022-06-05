import torch
from torch.utils.data.dataset import IterableDataset
from data.rl_data import Iterable_RL_Dataset
from data.torch_datasets import GeneralDataset, GeneralIterDataset
from load_objects import load_item
from accelerate import Accelerator
from utils.log_utils import DistributeCombineLogs, label_logs
from utils.misc import add_system_configs, convert_path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from utils.torch_utils import to
import random
import pickle as pkl
import os

def eval(cfg):
    print('using config:', cfg)
    eval_cfg = cfg['eval']
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])
    if eval_cfg['seed'] is not None:
        random.seed(eval_cfg['seed']+(torch.cuda.current_device() if torch.cuda.is_available() else 0))
        # random.seed(eval_cfg['seed'])
    
    raw_dataset = load_item(cfg['dataset'], system_cfg['device'])
    if isinstance(raw_dataset, Iterable_RL_Dataset):
        dataset = GeneralIterDataset(raw_dataset, 'cpu')
    else:
        dataset = GeneralDataset(raw_dataset, 'cpu')
    data_loader_kwargs = {'num_workers': eval_cfg['dataloader_workers'], 
                          'batch_size': eval_cfg['bsize'], 
                          'collate_fn': dataset.collate}
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = True
    data_loader = DataLoader(dataset, **data_loader_kwargs)

    evaluator = None
    if cfg['evaluator'] is not None:
        evaluator = load_item(cfg['evaluator'], system_cfg['device'])

    model = load_item(cfg['model'], system_cfg['device'])

    if isinstance(dataset, IterableDataset):
        model = accelerator.prepare(model)
    else:
        model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()

    eval_logs = DistributeCombineLogs(accelerator, use_wandb=False)
    with torch.no_grad():
        for i, eval_items in tqdm(enumerate(data_loader)):
            eval_items = to(eval_items, system_cfg['device'])
            if i >= eval_cfg['batches']:
                break
            _, logs, postproc_fs = accelerator.unwrap_model(model).get_loss(eval_items, **eval_cfg['loss'])
            if evaluator is not None:
                evaluator_logs = evaluator.evaluate(accelerator.unwrap_model(model), eval_items)
                if evaluator_logs is not None:
                    logs['evaluation'] = evaluator_logs
            eval_logs.accum_logs(logs)
            if (i + 1) % eval_cfg['print_every'] == 0:
                eval_total_logs = eval_logs.log(*postproc_fs, 
                                    partial(label_logs, label='eval'), 
                                    n=(i+1)*eval_cfg['bsize']*system_cfg['num_processes'])
    eval_total_logs = eval_logs.log(*postproc_fs, 
                                    partial(label_logs, label='eval'), 
                                    n=(i+1)*eval_cfg['bsize']*system_cfg['num_processes'])
    evaluator_dump = evaluator.dump()
    if eval_cfg['log_save_path'] is not None:
        if not os.path.exists(convert_path(os.path.dirname(eval_cfg['log_save_path']))):
            os.makedirs(convert_path(os.path.dirname(eval_cfg['log_save_path'])))
        with open(convert_path(eval_cfg['log_save_path']), 'wb') as f:
            pkl.dump({'all_logs': eval_total_logs, 'eval_dump': evaluator_dump, 'config': cfg}, f)

