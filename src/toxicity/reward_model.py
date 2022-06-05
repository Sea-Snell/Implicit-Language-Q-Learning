from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from models.base import BaseModel, InputType
from toxicity.toxicity_dataset import ToxicityListDataset
from abc import abstractmethod
from utils.torch_utils import to
from utils.misc import strip_from_beginning, strip_from_end

class RewardModel(BaseModel):
    def __init__(self, 
                 dataset: ToxicityListDataset, 
                 device: Union[torch.device, str]) -> None:
        super().__init__(dataset, device)
    
    @abstractmethod
    def get_reward_raw(self, items: InputType) -> List[float]:
        pass

    @abstractmethod
    def get_reward_str(self, text: Union[List[str],str]) -> Union[float, List[float]]:
        pass

class RobertaBinaryRewardModel(RewardModel):
    def __init__(self, 
                 dataset: ToxicityListDataset, 
                 device: Union[torch.device, str], 
                 roberta_kind: str, 
                 freeze_roberta: bool=True, 
                 reward_cuttoff: float=0.0):
        super().__init__(dataset, device)
        self.model = RobertaModel.from_pretrained(roberta_kind)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_kind)
        self.freeze_roberta = freeze_roberta
        self.reward_cuttoff = reward_cuttoff
        self.layers = nn.Sequential(
                        nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size*2), 
                        nn.ReLU(), 
                        nn.Linear(self.model.config.hidden_size*2, 1), 
                      )
    
    def prepare_inputs(self, items: InputType):
        items = super().prepare_inputs(items)
        raw_strs = self.dataset.tokenizer.decode(items['tokens'].detach().cpu().tolist())
        raw_strs = [strip_from_end(strip_from_beginning((raw_str[:raw_str.find('<|pad|>')] if raw_str.find('<|pad|>') != -1 else raw_str).strip(), '<a>'), '</a> </eod>').strip() for raw_str in raw_strs]
        new_tokenized = self.roberta_tokenizer(raw_strs, padding=True)
        items['tokens'] = torch.tensor(new_tokenized['input_ids']).to(self.device)
        items['attn_mask'] = torch.tensor(new_tokenized['attention_mask']).to(self.device)
        return items

    def forward(self, 
                tokens: torch.Tensor, 
                attn_mask: torch.Tensor):
        if not self.freeze_roberta:
            roberta_state = self.model(tokens, attention_mask=attn_mask).pooler_output
        else:
            with torch.no_grad():
                roberta_state = self.model(tokens, attention_mask=attn_mask).pooler_output
        predictions = self.layers(roberta_state)
        return F.sigmoid(predictions).squeeze(1)
    
    def get_reward_raw(self, 
                       items: InputType):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        predictions = self(tokens, attn_mask)
        return (predictions >= 0.5).float().detach().cpu().tolist()
    
    def get_reward_str(self, 
                       text: Union[List[str],str]):
        if isinstance(text, str):
            in_text = [text]
        else:
            in_text = text
        tokenized = self.roberta_tokenizer(in_text, padding=True)
        tokens, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
        tokens = torch.tensor(tokens).to(self.device)
        attn_mask = torch.tensor(attn_mask).to(self.device)
        tokens, attn_mask = tokens[:, :self.max_len], attn_mask[:, :self.max_len]
        predictions = self(tokens, attn_mask)
        outputs = (predictions >= 0.5).float()
        if isinstance(text, str):
            return outputs.squeeze(0).item()
        return outputs.detach().cpu().list()

    def get_loss(self, 
                 items: InputType):
        prepared_inputs = self.prepare_inputs(items)
        tokens, attn_mask = prepared_inputs['tokens'], prepared_inputs['attn_mask']
        targets = (prepared_inputs['rewards'].sum(dim=1) > self.reward_cuttoff).float()
        predictions = self(tokens, attn_mask)
        loss = F.binary_cross_entropy(predictions, targets)
        acc = ((predictions >= 0.5).float() == targets).float().mean()
        logs = {}
        n = tokens.shape[0]
        logs['loss'] = (loss.item(), n)
        logs['acc'] = (acc.item(), n)
        return loss, logs, []

