# Author: Yiran 'Lawrence' Luo
# Based on DomainBed (https://github.com/facebookresearch/DomainBed) 
# and MIRO (https://github.com/kakaobrain/miro)

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed import algorithms
from domainbed.algorithms import Algorithm

from .algorithms import *
from sconf import Config

def jsd(p, q):
    m = 0.5 * (p + q)
    loss = 0.0
    loss += F.kl_div(p, m, reduction="mean") 
    loss += F.kl_div(q, m, reduction="mean") 
 
    return loss * 0.5

class ForwardModel(nn.Module):
    """Forward model is used to reduce gpu memory usage of SWAD.
    """
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)


class MeanEncoder(nn.Module):
    """Identity function"""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x


class VarianceEncoder(nn.Module):
    """Bias-only model with diagonal covariance"""
    def __init__(self, shape, init=0.1, channelwise=True, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps

        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                # [B, C, H, W]
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                # CLIP-ViT: [H*W+1, B, C]
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()

        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


def get_shapes(model, input_shape, pretrained_prefeat=False):
    # get shape of intermediate features
    with torch.no_grad():
        dummy = torch.rand(1, *input_shape).to(next(model.parameters()).device)
        if pretrained_prefeat:
            feats = model(dummy)
        else:
            _, feats = model(dummy, ret_feats=True)
        shapes = [f.shape for f in feats]
        # print("prefeat shapes", shapes)

    return shapes

def freeze_(model):
    """Freeze model
    Note that this function does not control BN
    """
    for p in model.parameters():
        p.requires_grad_(False)

def get_pretrained_pre_featurizer(ckpt_path, input_shape, hparams):
    # ckpt_path = hparams['miro_pretrained_path']
    print("Loading pretrained pre-featurizer from", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model_dict']
    checkpoint_args = checkpoint['args']
    checkpoint_algorithm_name = checkpoint_args['algorithm']
    checkpoint_hparams = Config(checkpoint['model_hparams'])
    
    checkpoint_num_domains = len(checkpoint_args['test_envs']) - len(checkpoint_args['test_envs'][0])
    
    pre_featurizer_algorithm_class = algorithms.get_algorithm_class(checkpoint_algorithm_name)
    pre_featurizer_algorithm = pre_featurizer_algorithm_class(
        input_shape,
        1,
        checkpoint_num_domains,
        checkpoint_hparams,
    )
    
    linear_keywords = ["classifier", "network.1", "network.module.1","network_c", "network_s", "discriminator", "class_embeddings"]
        
    for k in list(state_dict.keys()):
        for linear_keyword in linear_keywords:
            if linear_keyword in k:
                del state_dict[k]
                break

    # print("# of pre-feat layers ditched: " + str(num_original_layers - len(state_dict)))
    msg = pre_featurizer_algorithm.load_state_dict(state_dict, strict=False)
    
    device_name = torch.cuda.get_device_name()
    
    # if 'A100' in device_name:
    if torch.cuda.device_count() == 1:
    
        if checkpoint_algorithm_name in ["ERM"]:
            pre_featurizer = pre_featurizer_algorithm.network[0]
        elif checkpoint_algorithm_name in ["ARM", "IRM", "MIRO"]:
            pre_featurizer = pre_featurizer_algorithm.network[0]
        elif checkpoint_algorithm_name in ["DANN", "CDANN"]:
            pre_featurizer = pre_featurizer_algorithm.featurizer
        elif checkpoint_algorithm_name in ["SagNet"]:
            pre_featurizer = pre_featurizer_algorithm.network_f
            
    else:
        if checkpoint_algorithm_name in ["ERM"]:
            pre_featurizer = pre_featurizer_algorithm.network.module[0]
        elif checkpoint_algorithm_name in ["ARM", "IRM", "MIRO"]:
            pre_featurizer = pre_featurizer_algorithm.network.module[0]
        elif checkpoint_algorithm_name in ["DANN", "CDANN"]:
            pre_featurizer = pre_featurizer_algorithm.featurizer.module
        elif checkpoint_algorithm_name in ["SagNet"]:
            pre_featurizer = pre_featurizer_algorithm.network_f.module
        
    freeze_(pre_featurizer)
    return pre_featurizer


class SMOSBarebone(Algorithm):
    """Scene-grounded Minimal dOmain Shift, adding the precursor loader"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        
        # Loading precursor teacher's featurizer
        if hparams['smos_pre_featurizer_pretrained'] :
            if "places" in hparams['smos_pre_featurizer_path']:
                tmp_model = self.hparams['model']
                self.hparams['model'] = "resnet50_places"
                self.pre_featurizer = URFeaturizer(
                    input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
                )
                self.hparams['model'] = tmp_model
                
            else:
                self.pre_featurizer = URFeaturizer(
                    input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
                )
            
                _pre_featurizer_pretrained = get_pretrained_pre_featurizer(
                    hparams['smos_pre_featurizer_path'], input_shape, hparams
                )
                self.pre_featurizer.load_state_dict(_pre_featurizer_pretrained.state_dict())
        else:
            self.pre_featurizer = URFeaturizer(
                input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
            )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)
            
        self.ld = hparams.ld
        self.ld_KL = hparams.ld_KL

        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([
            MeanEncoder(shape) for shape in shapes
        ])
        self.var_encoders = nn.ModuleList([
            VarianceEncoder(shape) for shape in shapes
        ])
        
        if torch.cuda.device_count() > 1:
            print('Using multi-gpu for model paralleling')
            self.network = nn.DataParallel(self.network)
            # self.pre_featurizer = nn.DataParallel(self.pre_featurizer)
            

        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {"params": self.mean_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
            {"params": self.var_encoders.parameters(), "lr": hparams.lr * hparams.lr_mult},
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model


class SMOS_JS(SMOSBarebone):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
    
    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        with torch.no_grad():
            pre_feats_out, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.
        
        loss += reg_loss * self.ld_KL / self.hparams['lr_mult']
        
        feat_loss = jsd(pre_feats_out, feat)
        
        loss += feat_loss * (self.ld_KL)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), 'feat_loss': feat_loss.item()}

class SMOS_KL(SMOSBarebone):
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
    
    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        with torch.no_grad():
            pre_feats_out, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.
        
        loss += reg_loss * self.ld_KL / self.hparams['lr_mult']
        
        feat_loss = 0.
        feat_loss = F.kl_div(pre_feats_out, feat, reduction="mean") 
        
        loss += feat_loss * (self.ld_KL)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), 'feat_loss': feat_loss.item()}