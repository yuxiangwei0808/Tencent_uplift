import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from .abs_mt_arch import AbsArchitecture


class MMOE(AbsArchitecture):
    def __init__(self, encoder_class, num_experts, task_names, enc_kwargs, in_feat, rep_grad, **kwargs):
        r"""
        a Multi-gate MOE to encoder features
        Args:
            encoder_class (nn.Module, list): encoder class of each expert. Can be a neural net or another MOE
            num_experts (int): number of experts
            task_names (list): names of tasks
            enc_kwargs (dict): dict of encoder kwargs that corresponds to each task
            in_feat (int): input feature number
        """
        super().__init__(task_names, encoder_class, None, rep_grad, **kwargs)
        self.num_experts = num_experts
        self.task_names = task_names
        assert len(task_names) == num_experts, "each task should have an expert"

        if len(enc_kwargs) > 1:
            # each expert will have separate kwargs
            assert len(enc_kwargs) == num_experts == len(encoder_class), "each expert should have corresponding args"
            self.expert_shared = nn.ModuleList([enc(**arg) for enc, arg in zip(encoder_class, enc_kwargs.values())])
        else:
            self.expert_shared = nn.ModuleList([encoder_class(**enc_kwargs['all']) for _ in range(num_experts)])

        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(in_feat, self.num_experts),
                                                    nn.Softmax(dim=-1)) for task in self.task_name})
        # self.decoders = nn.ModuleDict({task: nn.Linear(rep_dim, 1) for task in self.task_names})
        
    def forward(self, inputs: torch.tensor, task_name : str =None):
        # TODO use nn.embedding to encode inputs as EFIN
        experts_shared_rep = torch.stack([e(inputs) for e in self.expert_shared])
            
        out = {}
        for task in self.task_names:
            if task_name is not None and task != task_name:
                continue

            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1))
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = gate_rep
            # out[task] = self.decoders[task](gate_rep)
        return out
    
    def get_share_params(self):
        return self.expert_shared.parameters()

    def zero_grad_share_params(self):
        self.expert_shared.zero_grad(set_to_none=False)