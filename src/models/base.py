import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Dict, Optional, Tuple, Union, List
from data.rl_data import DataPoint, RL_Dataset
from utils.torch_utils import to

InputType = Union[List[DataPoint], Dict[str, Union[torch.Tensor, Any]]]

class BaseModel(ABC, nn.Module):
    def __init__(self, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str]) -> None:
        super().__init__()
        self.dataset = dataset
        self.device = device
        self.max_len = self.dataset.max_len

    def prepare_inputs(self, items: InputType):
        if isinstance(items, dict):
            return items
        return to(self.dataset.collate(items, self.device), self.device)

    @abstractmethod
    def get_loss(self, items: InputType, **kwargs):
        pass

class BaseTransformer(BaseModel):
    def __init__(self, 
                 pretrained_model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str]) -> None:
        super().__init__(dataset, device)
        self.model = pretrained_model
        self.model.resize_token_embeddings(self.dataset.tokenizer.num_tokens())

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: BaseModel, items: InputType) -> Optional[Dict[str, Any]]:
        pass

    def postproc(self, logs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return logs
    
    def dump(self) -> Any:
        return None

