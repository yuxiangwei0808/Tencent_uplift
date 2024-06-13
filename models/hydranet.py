import torch
import numpy as np
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet


class BatchNormWeighted(nn.BatchNorm2d):
    def __init__(self, num_features, wts):
        super(BatchNormWeighted, self).__init__(num_features)
        self.wts = wts

    def forward(self, x):
        self._check_input_dim(x)
        self.wts = self.wts.cuda(x.device)
        flat_x = x.view(x.size(0), -1)
        weighted_x = (flat_x * self.wts[:flat_x.shape[0], None]) / self.wts[:flat_x.shape[0]].sum()
        weighted_x = weighted_x.view(x.shape).permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
        y = x.permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)
        
        mu = weighted_x.mean(dim=1)
        sigma2 = weighted_x.var(dim=1)
        
        if self.training:
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * sigma2
            y = (y - mu.view(-1, 1)) / (sigma2.view(-1, 1).sqrt() + self.eps)
        else:
            y = (y - self.running_mean.view(-1, 1)) / (self.running_var.view(-1, 1).sqrt() + self.eps)
        
        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view_as(x).permute(1, 0, 2, 3)


def makeBNWeighted(net_, wts):
    for name, child in net_.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(net_, name, BatchNormWeighted(child.num_features, wts=wts))
        makeBNWeighted(child, wts)


class Uncoiler(nn.Module):
    def __init__(self, net_):
        super(Uncoiler, self).__init__()
        mod_dict = self.get_module_dict(net_)
        for key, module in mod_dict.items():
            self.add_module(key, module)

    def get_module_dict(self, net_):
        mod_dict = {}
        for name, child in net_.named_children():
            if len(list(child.children())) == 0 or 'trans' in name:
                mod_dict[name] = child
            else:
                for name1, child1 in child.named_children():
                    mod_dict[f"{name}_{name1}"] = child1
        return mod_dict

    def forward(self, x):
        for i, mod in enumerate(self.children(), 1):
            x = mod(x)
            if i == len(list(self.children())):
                x = torch.nn.functional.avg_pool2d(x, x.size()[3])
                x = x.view(x.size(0), -1)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class HydraNet(nn.Module):
    def __init__(self, model, n_heads=1, split_pt=7, num_classes=10, batch_size=128, sample_wts=None, path=None):
        super(HydraNet, self).__init__()
        self.n_heads = n_heads
        self.split_pt = split_pt

        if sample_wts is None:
            sample_wts = [Dirichlet(torch.ones(batch_size)).sample().float().cuda() for _ in range(n_heads)]
        self.sample_wts = sample_wts
      
    
        model_body = self.model_maker(model, num_classes)
        if path:
            model_body.load_state_dict(torch.load(path)['state_dict'])
        
        model_body = Uncoiler(model_body)
        model_body = Uncoiler(model_body)
        self.layer_1 = nn.Sequential(*list(model_body.children())[:split_pt])
        
        self.layer_2 = nn.ModuleList()
        for i in range(self.n_heads):
            model_head = self.model_maker(model, num_classes)
            model_head = Uncoiler(model_head)
            model_head = Uncoiler(model_head)
            makeBNWeighted(model_head, self.sample_wts[i])
            modules = list(model_head.children())[split_pt:-1]
            self.layer_2.append(nn.Sequential(*modules, Flatten(), list(model_head.children())[-1]))

    def model_maker(self, model, num_classes):
        if model == "resnet":
            model_ = eval(model)(depth=110, num_classes=num_classes).cuda()
        elif model == "preresnet":
            model_ = eval(model)(depth=110, num_classes=num_classes).cuda()
        elif model == "densenet":
            model_ = eval(model)(depth=100, num_classes=num_classes, growthRate=12).cuda()
        else:
            raise ValueError("Unrecognized model name given")
        return model_

    def forward(self, x):
        x = self.layer_1(x)
        outputs = [head(x) for head in self.layer_2]
        return outputs 