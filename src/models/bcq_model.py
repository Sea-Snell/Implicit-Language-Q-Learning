from typing import Optional, Union
from transformers import PreTrainedModel
from data.rl_data import RL_Dataset
from models.base import InputType
from models.cql_model import CQLModel
import torch
import torch.nn.functional as F
import wandb
from utils.torch_utils import get_transformer_logs

class BCQModel(CQLModel):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 alpha: float = 0.005, 
                 gamma=1.0, 
                 beta=1.0, 
                 transition_weight=0.0, 
                 clip_weight: Optional[float] = None, 
                 value_max: Optional[float] = None, 
                 value_min: Optional[float] = None, 
                 detach_v: bool = False, 
                 detach_pi: bool = False, 
                 detach_q: bool = False, 
                 double_q: bool = False, 
                 seperate_policy: bool = False, 
                 seperate_target: bool = False, 
                 exp_weights: bool = False, 
                 advanced_mlp: bool = False, 
                 cql_temp: float = 1.0, 
                ):
        super().__init__(model, dataset, device, alpha, gamma, beta, transition_weight, 
                         clip_weight, value_max, value_min, detach_v=detach_v, detach_pi=detach_pi, 
                         detach_q=detach_q, double_q=double_q, seperate_policy=seperate_policy, 
                         seperate_target=seperate_target, exp_weights=exp_weights, advanced_mlp=advanced_mlp, 
                         cql_temp=cql_temp)
    
    def get_q_loss(self, all_target_qns, qs, rs, log_probs, gamma, r_scale, terminals):
        all_target_qns = all_target_qns.detach()
        log_probs = log_probs.detach()
        max_qns = torch.max(all_target_qns, dim=-1).values
        if self.double_q:
            q1, q2 = qs
            l1 = (F.huber_loss(q1, (1 - terminals[:, 1:]) * max_qns * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            l2 = (F.huber_loss(q2, (1 - terminals[:, 1:]) * max_qns * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            return l1 + l2
        return (F.huber_loss(qs, (1 - terminals[:, 1:]) * max_qns * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)

    def get_loss(self, 
                 items: InputType, 
                 awac_weight=0.0, 
                 v_loss_weight=0.0, 
                 q_loss_weight=0.0, 
                 cql_loss_weight=0.0, 
                 dm_loss_weight=0.0, 
                 mc_returns=False, 
                 r_scale=1.0):
        prepared_inputs = self.prepare_inputs(items)
        a_idx = prepared_inputs['action_idxs']
        get_qvs_outputs = self.get_qvs(items, 
                                       qv_kwargs={'output_attentions': True}, 
                                       policy_kwargs={'output_attentions': True}, 
                                       target_kwargs={'output_attentions': True}, 
                                       detach_full_policy=True, 
                                      )
        tokens, attn_mask, model_outputs = get_qvs_outputs['tokens'], get_qvs_outputs['attn_mask'], get_qvs_outputs['model_outputs']
        vs, qs = get_qvs_outputs['vs'], get_qvs_outputs['qs']
        vns, target_qs, rs = get_qvs_outputs['vns'], get_qvs_outputs['target_qs'], get_qvs_outputs['rs']
        terminals, logits, weights = get_qvs_outputs['terminals'], get_qvs_outputs['logits'], get_qvs_outputs['weights']
        all_target_qns = get_qvs_outputs['all_target_qns']

        logs = {}
        transformer_logs = {}
        transformer_logs['qv_transformer_logs'] = get_transformer_logs(model_outputs['qv_model_outputs'].attentions, self.model, attn_mask)
        if self.lm_policy is not None:
            transformer_logs['policy_transformer_logs'] = get_transformer_logs(model_outputs['policy_model_outputs'].attentions, self.lm_policy, attn_mask)
        if self.lm_target is not None:
            transformer_logs['target_transformer_logs'] = get_transformer_logs(model_outputs['target_model_outputs'].attentions, self.lm_target, attn_mask)
        n = max((1 - terminals[:, :-1]).sum().item(), 1)
        rs_downstream = self.get_downstream_rs(rs, self.gamma)
        if mc_returns:
            v_loss = self.get_v_loss(vs, rs_downstream, terminals)
        else:
            v_loss = self.get_v_loss(vs, target_qs, terminals)
        select_tokens = torch.gather(tokens[:, 1:], dim=1, index=a_idx)
        select_logprobs = torch.gather(torch.log_softmax(logits[:, :-1, :], dim=2), dim=1, index=a_idx.unsqueeze(2).repeat(1, 1, logits.shape[2]))
        log_probs = torch.gather(select_logprobs, dim=2, index=select_tokens.unsqueeze(2)).squeeze(2)
        q_loss = self.get_q_loss(all_target_qns, qs, rs, log_probs, self.gamma, r_scale, terminals)
        cql_loss = get_qvs_outputs['cql_term']
        dm_loss = get_qvs_outputs['dm_term']
        token_loss = self.awac_loss(tokens, attn_mask, logits, weights)
        logs['token_loss'] = (token_loss.item(), n)
        loss = awac_weight * token_loss + v_loss_weight * v_loss + q_loss_weight * q_loss + cql_loss_weight * cql_loss + dm_loss_weight * dm_loss
        logs['v_loss'] = (v_loss.item(), n)
        logs['q_loss'] = (q_loss.item(), n)
        logs['cql_loss'] = (cql_loss.item(), n)
        logs['dm_loss'] = (dm_loss.item(), n)
        advantages = sum([((target_qs[i] - vs[i])[:(1 - terminals[i, :-1]).sum().long().item()]).detach().cpu().tolist() for i in range(tokens.shape[0])], [])
        if self.double_q:
            q1, q2 = qs
            logs['q1_avg'] = ((q1 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q1_var'] = (((((q1 - logs['q1_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
            logs['q2_avg'] = ((q2 * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q2_var'] = (((((q2 - logs['q2_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        else:
            logs['q_avg'] = ((qs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
            logs['q_var'] = (((((qs - logs['q_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        logs['v_avg'] = ((vs * (1 - terminals[:, :-1])).sum().item() / max(n, 1), n)
        logs['v_var'] = (((((vs - logs['v_avg'][0]) ** 2)*(1 - terminals[:, :-1])).sum() / max(n, 1)).item(), 1)
        act_weights = torch.gather(weights, dim=1, index=a_idx)
        logs['act_weight_avg'] = (((act_weights * (1 - terminals[:, :-1])).sum() / max(n, 1)).item(), n)
        logs['transformer'] = transformer_logs
        postproc_f = lambda l: l.update({'loss': awac_weight * l['token_loss'] + q_loss_weight * l['q_loss'] + v_loss_weight * l['v_loss'] + cql_loss_weight * l['cql_loss'] + dm_loss_weight * l['dm_loss']})
        hist_f = lambda l: l.update({'advantage_hist': wandb.Histogram(advantages)})
        return loss, logs, [postproc_f, hist_f]

class PsiModel(BCQModel):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 alpha: float = 0.005, 
                 gamma=1.0, 
                 beta=1.0, 
                 transition_weight=0.0, 
                 clip_weight: Optional[float] = None, 
                 value_max: Optional[float] = None, 
                 value_min: Optional[float] = None, 
                 detach_v: bool = False, 
                 detach_pi: bool = False, 
                 detach_q: bool = False, 
                 double_q: bool = False, 
                 seperate_policy: bool = False, 
                 seperate_target: bool = False, 
                 exp_weights: bool = False, 
                 advanced_mlp: bool = False, 
                 cql_temp: float = 1.0, 
                ):
        super().__init__(model, dataset, device, alpha, gamma, beta, transition_weight, 
                         clip_weight, value_max, value_min, detach_v=detach_v, detach_pi=detach_pi, 
                         detach_q=detach_q, double_q=double_q, seperate_policy=seperate_policy, 
                         seperate_target=seperate_target, exp_weights=exp_weights, advanced_mlp=advanced_mlp, 
                         cql_temp=cql_temp)
    

    def get_q_loss(self, all_target_qns, qs, rs, log_probs, gamma, r_scale, terminals):
        all_target_qns = all_target_qns.detach()
        log_probs = log_probs.detach()
        log_sum_exp_targets = torch.logsumexp(all_target_qns, dim=-1)
        if self.double_q:
            q1, q2 = qs
            l1 = (F.huber_loss(q1, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            l2 = (F.huber_loss(q2, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            return l1 + l2
        return (F.huber_loss(qs, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
    
class GModel(BCQModel):
    def __init__(self, 
                 model: PreTrainedModel, 
                 dataset: RL_Dataset, 
                 device: Union[torch.device, str] = "cuda", 
                 alpha: float = 0.005, 
                 gamma=1.0, 
                 beta=1.0, 
                 transition_weight=0.0, 
                 clip_weight: Optional[float] = None, 
                 value_max: Optional[float] = None, 
                 value_min: Optional[float] = None, 
                 detach_v: bool = False, 
                 detach_pi: bool = False, 
                 detach_q: bool = False, 
                 double_q: bool = False, 
                 seperate_policy: bool = False, 
                 seperate_target: bool = False, 
                 exp_weights: bool = False, 
                 advanced_mlp: bool = False, 
                 cql_temp: float = 1.0, 
                ):
        super().__init__(model, dataset, device, alpha, gamma, beta, transition_weight, 
                         clip_weight, value_max, value_min, detach_v=detach_v, detach_pi=detach_pi, 
                         detach_q=detach_q, double_q=double_q, seperate_policy=seperate_policy, 
                         seperate_target=seperate_target, exp_weights=exp_weights, advanced_mlp=advanced_mlp, 
                         cql_temp=cql_temp)
    

    def get_q_loss(self, all_target_qns, qs, rs, log_probs, gamma, r_scale, terminals):
        all_target_qns = all_target_qns.detach()
        log_probs = log_probs.detach()
        log_sum_exp_targets = torch.logsumexp(all_target_qns + log_probs, dim=-1)
        if self.double_q:
            q1, q2 = qs
            l1 = (F.huber_loss(q1, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            l2 = (F.huber_loss(q2, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
            return l1 + l2
        return (F.huber_loss(qs, (1 - terminals[:, 1:]) * log_sum_exp_targets * gamma + (rs / r_scale) + log_probs, reduction='none') * (1 - terminals[:, :-1])).sum() / max((1 - terminals[:, :-1]).sum().item(), 1.0)
