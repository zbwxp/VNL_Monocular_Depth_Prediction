import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import lateral_net as lateral_net
from lib.utils.net_tools import get_func
from lib.models.WCEL_loss import WCEL_Loss
from lib.models.VNL_loss import VNL_Loss
from lib.models.TV_loss import TV_Loss
from lib.models.image_transfer import bins_to_depth, kitti_merge_imgs
from lib.core.config import cfg
from .mix_transformer import mit_b2
from .nrd_trans_stage5 import nrd_trans_stage5

class MetricDepthModel(nn.Module):
    def __init__(self):
        super(MetricDepthModel, self).__init__()
        self.loss_names = ['Weighted_Cross_Entropy', 'Virtual_Normal']
        self.depth_model = DepthModel_nrd()

    def forward(self, data):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.a_real = data['A'].cuda()
        self.b_fake_logit, self.b_fake_softmax = self.depth_model(self.a_real)
        return {'b_fake_logit': self.b_fake_logit, 'b_fake_softmax': self.b_fake_softmax}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)

            if cfg.MODEL.PREDICTION_METHOD == 'classification':
                pred_depth = bins_to_depth(out['b_fake_softmax'])
            elif cfg.MODEL.PREDICTION_METHOD == 'regression':
                # for regression methods
                pred_depth = torch.nn.functional.sigmoid(out['b_fake_logit'])
            else:
                raise ValueError("Unknown prediction methods")

            out = pred_depth
            return {'b_fake': out}

    def inference_kitti(self, data):
        #crop kitti images into 3 parts
        with torch.no_grad():
            self.a_l_real = data['A_l'].cuda()
            _, b_l_classes = self.depth_model(self.a_l_real)
            self.b_l_fake = bins_to_depth(b_l_classes)

            self.a_m_real = data['A_m'].cuda()
            _, b_m_classes = self.depth_model(self.a_m_real)
            self.b_m_fake = bins_to_depth(b_m_classes)

            self.a_r_real = data['A_r'].cuda()
            _, b_r_classes = self.depth_model(self.a_r_real)
            self.b_r_fake = bins_to_depth(b_r_classes)

            out = kitti_merge_imgs(self.b_l_fake, self.b_m_fake, self.b_r_fake, torch.squeeze(data['B_raw']).shape, data['crop_lmr'])
            return {'b_fake': out}


class ModelLoss(object):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.weight_cross_entropy_loss = WCEL_Loss()
        self.virtual_normal_loss = VNL_Loss(focal_x=cfg.DATASET.FOCAL_X, focal_y=cfg.DATASET.FOCAL_Y, input_size=cfg.DATASET.CROP_SIZE)
        self.smooth_loss = TV_Loss()

    def criterion(self, pred_softmax, pred_logit, data, epoch):
        pred_depth = bins_to_depth(pred_softmax)
        loss_metric = self.weight_cross_entropy_loss(pred_logit, data['B_bins'], data['B'].cuda())
        loss_normal = self.virtual_normal_loss(data['B'].cuda(), pred_depth)
        loss_smooth = self.smooth_loss(data['B'].cuda(), pred_depth)

        loss = {}
        loss['metric_loss'] = loss_metric
        loss['virtual_normal_loss'] = cfg.MODEL.DIFF_LOSS_WEIGHT * loss_normal
        loss['conditional_smooth_vt_loss'] = loss_smooth
        loss['total_loss'] = loss['metric_loss'] + loss['virtual_normal_loss'] + loss['conditional_smooth_vt_loss']
        return loss


class ModelOptimizer(object):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.0005

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
            ]
        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)
    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        self.optimizer.step()

class ModelOptimizer_adamw(object):
    def __init__(self, model):
        super(ModelOptimizer_adamw, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'encoder_modules' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = cfg.TRAIN.BASE_LR
        lr_decoder = cfg.TRAIN.BASE_LR * cfg.TRAIN.SCALE_DECODER_LR
        weight_decay = 0.01
        betas = (0.9, 0.999)

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'betas': betas,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'betas': betas,
             'weight_decay': weight_decay},
            ]
        self.optimizer = torch.optim.AdamW(net_params)
    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        self.optimizer.step()


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        bottom_up_model = 'lateral_net.lateral_' + cfg.MODEL.ENCODER
        self.encoder_modules = get_func(bottom_up_model)()
        self.decoder_modules = lateral_net.fcn_topdown(cfg.MODEL.ENCODER)

    def forward(self, x):
        lateral_out, encoder_stage_size = self.encoder_modules(x)
        out_logit, out_softmax = self.decoder_modules(lateral_out, encoder_stage_size)
        return out_logit, out_softmax

class DepthModel_nrd(nn.Module):
    def __init__(self):
        super(DepthModel_nrd, self).__init__()
        self.encoder_modules = mit_b2()
        if self.training:
            self.encoder_modules.init_weights(pretrained='/media/bz/D/美团/VNL_Monocular_Depth_Prediction/pretrained/mit_b2.pth')
        self.decoder_modules = nrd_trans_stage5()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        b, c, h, w = x.size()
        features = self.encoder_modules(x)
        out_logit = self.decoder_modules(features)
        out_logit = F.interpolate(out_logit, size=(h, w), mode='bilinear', align_corners=False)

        return out_logit, self.softmax(out_logit)

def cal_params(model):
    model_dict = model.state_dict()
    paras = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum = 0

    for key in model_dict.keys():
        print(key)
        if 'layer5' not in key:
            if 'running' not in key:
                print(key)
                ss = model_dict[key].size()
                temp = 1
                for s in ss:
                    temp = temp * s
                print(temp)
                sum = sum + temp
    print(sum)
    print(paras)
