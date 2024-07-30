import torch
import torch.nn as nn
import math
import sys

do_rate = 0.1
normalization = 'device'

def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def sigmod2(y):
    # y = torch.clamp(0.995 / (1.0 + torch.exp(-y)) + 0.0025, 0, 1)
    # y = torch.clamp(y, -16, 16)
    y=torch.sigmoid(y)
    # y = 0.995 / (1.0 + torch.exp(-y)) + 0.0025

    return y

def safe_sqrt(x):
    ''' Numerically safe version of Pytoch sqrt '''
    return torch.sqrt(torch.clip(x, 1e-9, 1e+9))

class ShareNetwork(nn.Module):
    def __init__(self, input_dim, share_dim, base_dim, device):
        super(ShareNetwork, self).__init__()
        if True:
            print("use BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, share_dim),
                # nn.BatchNorm1d(share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, base_dim),
                # nn.BatchNorm1d(base_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate)
            )
        else:
            print("No BatchNorm1d")
            self.DNN = nn.Sequential(
                nn.Linear(input_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, share_dim),
                nn.ELU(),
                nn.Dropout(p=do_rate),
                nn.Linear(share_dim, base_dim),
                nn.ELU(),
            )

        self.DNN.apply(init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        h_rep = self.DNN(x)
        if normalization == "divide":
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep), dim=1, keepdim=True))
        else:
            h_rep_norm = 1.0 * h_rep
        return h_rep_norm


class BaseModel(nn.Module):
    def __init__(self, base_dim, ):
        super(BaseModel, self).__init__()
        self.DNN = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate)
        )
        self.DNN.apply(init_weights)

    def forward(self, x):
        logits = self.DNN(x)
        return logits

class BaseModel4MetaLearner(nn.Module):
    def __init__(self, input_dim, base_dim, device):
        super(BaseModel4MetaLearner, self).__init__()
        self.DNN = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, base_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            # nn.ELU(),
            # nn.Dropout(p=.do_rate),
            # nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(share_dim),
            nn.ELU(),
            nn.Dropout(p=do_rate),
            nn.Linear(base_dim, 1),
            # nn.ELU()
            # nn.BatchNorm1d(base_dim),
        )
        self.DNN.apply(init_weights)
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        logit = self.DNN(x)
        return logit


class PrpsyNetwork(nn.Module):
    """propensity network"""
    def __init__(self, base_dim, ):
        super(PrpsyNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim, )
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


class Mu0Network(nn.Module):
    def __init__(self, base_dim, ):
        super(Mu0Network, self).__init__()
        self.baseModel = BaseModel(base_dim, )
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class Mu1Network(nn.Module):
    def __init__(self, base_dim, ):
        super(Mu1Network, self).__init__()
        self.baseModel = BaseModel(base_dim, )
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        # return self.relu(p)
        return p


class TauNetwork(nn.Module):
    """pseudo tau network"""
    def __init__(self, base_dim):
        super(TauNetwork, self).__init__()
        self.baseModel = BaseModel(base_dim)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.logitLayer.apply(init_weights)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        tau_logit = self.logitLayer(inputs)
        # return self.tanh(p)
        return tau_logit


class ESX(nn.Module):
    """ESX"""
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork, device):
        super(ESX, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)
        self.loss_fn = nn.BCELoss()
        self.loss_fn_logit = nn.BCEWithLogitsLoss()
        self.device = device
        self.to(device)

    def forward(self, inputs):
        shared_h = self.shareNetwork(inputs)

        # propensity output_logit
        p_prpsy_logit = self.prpsy_network(shared_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1 # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0 # Refer to the naming in TARnet/CFR


        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h
    
    def calculate_loss(self, input, t_true, y_true):
        if t_true.dim() != 2:
            t_true = t_true.unsqueeze(1)
        if y_true.dim() != 2:
            y_true = y_true.unsqueeze(1)
            
        p_prpsy_logit, p_estr, p_escr, p_tau_logit, p_mu1_logit, p_mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h = self.forward(input)
        prpsy_loss = 0.5 * self.loss_fn_logit(p_prpsy_logit, t_true)
        # loss for ESTR, ESCR
        estr_loss = 0.5 * self.loss_fn(p_estr, y_true * t_true)
        escr_loss = self.loss_fn(p_escr, y_true * (1 - t_true))

        #loss for TR, CR
        tr_loss = 0.001 * self.loss_fn(p_h1, y_true)  # * (1 / (2 * p_t))
        cr_loss = 0.001 * self.loss_fn(p_h0, y_true)  # * (1 / (2 * (1 - p_t)))
        
        cross_tr_loss = self.loss_fn(torch.sigmoid(p_mu0_logit + p_tau_logit), y_true)
        cross_cr_loss = 0.5 * self.loss_fn(torch.sigmoid(p_mu1_logit - p_tau_logit), y_true)
        return prpsy_loss + estr_loss + escr_loss + tr_loss + cr_loss + cross_tr_loss + cross_cr_loss

        
def DESCN(input_dim, base_dim=64, share_dim=128, device=torch.cuda.device()):
    share_network = ShareNetwork(input_dim=input_dim, share_dim=share_dim, base_dim=base_dim, device=device)
    prpsy_network = PrpsyNetwork(base_dim)
    mu1_network = Mu1Network(base_dim)
    mu0_network = Mu0Network(base_dim)
    tau_network = TauNetwork(base_dim)
    return ESX(prpsy_network, mu1_network, mu0_network, tau_network, share_network, device)