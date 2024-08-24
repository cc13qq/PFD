import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import init, Parameter

class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x): # (b, embsize)
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)
        

class xception_net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.fc = AngleSimpleLinear(2048, num_classes)
    def feature_map(self, x):
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)
        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)        
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)
        fea = x    
        return fea

    def classifier(self, fea_map):
        out, fea = self.xception_rgb.classifier(fea_map) # relu + avgpool
        return out, fea
    
    # def get_logit(self, feature):
    #     logits_cls = self.fc(feature) # (b, num_classes)
    #     return logits_cls

    def forward(self, x, return_feature=False):
        _, feature = self.classifier(self.feature_map(x)) # (b, 2048)
        logits_cls = self.fc(feature) # (b, num_classes)
        if return_feature:
            return logits_cls, feature
        else:
            return logits_cls

class AngleSimpleLinear_sep(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear_sep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # self.weight = torch.nn.Parameter(w.unsqueeze(0).repeat(batch_size,1,1))

    def forward(self, x): # (b, embsize, Hf, Wf)
        # cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))

        x = x.view(x.size(0), x.size(1), -1) # (b, embsize, Hf*Wf)

        cos_theta = torch.cat([F.normalize(x[:,:,i], dim=1).mm(F.normalize(self.weight, dim=0)).unsqueeze(-1) for i in range(x.size(-1))],-1) # (b, out_features, Hf*Wf)

        return cos_theta.clamp(-1, 1)

class xception_net_sep_pre(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, num_classes)
        # self.anglelinear = AngleSimpleLinear_sep(2048, num_classes)
       
        self.weight = Parameter(torch.Tensor(2048, num_classes)) # anglelinear weight
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def anglelinear(self, x): # AngleSimpleLinear_sep

        x = x.view(x.size(0), x.size(1), -1) # (b, embsize, Hf*Wf)

        cos_theta = torch.cat([F.normalize(x[:,:,i], dim=1).mm(F.normalize(self.weight, dim=0)).unsqueeze(-1) for i in range(x.size(-1))],-1) # (b, out_features, Hf*Wf)

        return cos_theta.clamp(-1, 1)
    
    def fc(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)

    def features(self, x):
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)
        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)        
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)  

        return x
    def feature_avg_pool(self, x):
        features = self.features(x)
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # (b, embsize)
        return x
    
    def Sep_classifier(self, fea_map): # (b, embsize, Hf, Wf)
        # feas = self.relu(features)
        # outs = torch.zeros([feas.shape[0], 2, feas.shape[-2], feas.shape[-1]]).to(features.device) # (b, num_class, Hf, Wf)

        outs = self.anglelinear(fea_map) # (b, num_class, Hf*Wf)
        outs = outs.view(outs.size(0), outs.size(1), fea_map.size(2), fea_map.size(3)) # (b, num_class, Hf, Wf)

        # x = feas.view(feas.size(0), feas.size(1), -1) # (b, embsize, Hf*Wf)

        # ts = torch.cat([tt[:,:,i].mm(w).unsqueeze(-1) for i in range(tt.size(-1))],-1)

        # for i in range(feas.shape[-2]):
        #     for j in range(feas.shape[-1]):
        #         x = feas[:,:,i,j].clone()
        #         x = x.view(x.size(0), -1) # (b, embsize)
        #         out_ij = self.anglelinear(x)
        #         outs[:,:,i,j] = out_ij.clone() # (b, num_class, Hf, Wf)

        out = F.adaptive_max_pool2d(outs, (1, 1))
        out = out.view(out.size(0), -1) # (b,num_class)
        return out
    
    def classifier(self, fea_map):
        x = self.relu(fea_map)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x
    
    # def get_logit(self, feature):
    #     logits_cls = self.Sep_classifier(feature) # (b, num_classes)
    #     return logits_cls

    def forward(self, x, return_feature=False):
        fea_map = self.features(x)
        fea_map = self.relu(fea_map)
        logits_cls = self.Sep_classifier(fea_map)
        if return_feature:
            # fea = self.relu(fea_map)
            feature = F.adaptive_avg_pool2d(fea_map, (1, 1))
            feature = feature.view(feature.size(0), -1)
            return logits_cls, feature
        else:
            return logits_cls


class xception_net_sep1(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, 1)
        # self.anglelinear = AngleSimpleLinear_sep(2048, num_classes)
       
        self.weight = Parameter(torch.Tensor(2048, 1)) # anglelinear weight
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def anglelinear(self, x): # AngleSimpleLinear_sep

        x = x.view(x.size(0), x.size(1), -1) # (b, embsize, Hf*Wf)

        cos_theta = torch.cat([F.normalize(x[:,:,i], dim=1).mm(F.normalize(self.weight, dim=0)).unsqueeze(-1) for i in range(x.size(-1))],-1) # (b, out_features, Hf*Wf)

        return cos_theta.clamp(-1, 1)
    
    def fc(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)

    def features(self, x):
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)
        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)        
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)  

        return x
    def feature_avg_pool(self, x):
        features = self.features(x)
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # (b, embsize)
        return x
    
    def Sep_classifier(self, fea_map): # (b, embsize, Hf, Wf)
        outs = self.anglelinear(fea_map) # (b, 1, Hf*Wf)
        outs = outs.view(outs.size(0), outs.size(1), fea_map.size(2), fea_map.size(3)) # (b, 1, Hf, Wf)

        out = F.adaptive_max_pool2d(outs, (1, 1))
        out = out.view(out.size(0), -1) # (b,1)
        return out
    
    def classifier(self, fea_map):
        x = self.relu(fea_map)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, x, return_feature=False):
        fea_map = self.features(x)
        fea_map = self.relu(fea_map)
        logits_cls = self.Sep_classifier(fea_map)
        logits_cls = logits_cls.squeeze(-1)
        if return_feature:
            # fea = self.relu(fea_map)
            feature = F.adaptive_avg_pool2d(fea_map, (1, 1))
            feature = feature.view(feature.size(0), -1)
            return logits_cls, feature
        else:
            return logits_cls
    

class xception_net_sep(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(2048, num_classes)
        # self.anglelinear = AngleSimpleLinear_sep(2048, num_classes)
       
        self.weight = Parameter(torch.Tensor(2048, num_classes)) # anglelinear weight
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def anglelinear(self, x): # AngleSimpleLinear_sep

        x = x.view(x.size(0), x.size(1), -1) # (b, embsize, Hf*Wf)

        cos_theta = torch.cat([F.normalize(x[:,:,i], dim=1).mm(F.normalize(self.weight, dim=0)).unsqueeze(-1) for i in range(x.size(-1))],-1) # (b, out_features, Hf*Wf)

        return cos_theta.clamp(-1, 1)
    
    def fc(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)

    def features(self, x):
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)
        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)        
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)  

        return x
    def feature_avg_pool(self, x):
        features = self.features(x)
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # (b, embsize)
        return x
    
    def Sep_classifier(self, fea_map): # (b, embsize, Hf, Wf)
        outs = self.anglelinear(fea_map) # (b, 1, Hf*Wf)
        outs = outs.view(outs.size(0), outs.size(1), fea_map.size(2), fea_map.size(3)) # (b, 1, Hf, Wf)

        _, idxs = F.adaptive_max_pool2d(-outs[:,1,:,:], (1, 1), return_indices=True) # (b,1,1,1) 对第二维求maxpooling并返回indexs
        Hf, Wf = outs.shape[2], outs.shape[3]
        idxs_i, idxs_j = idxs/Wf, idxs%Wf
        idxs_ij = torch.cat([idxs_i.long().unsqueeze(1), idxs_j.long().unsqueeze(1)], dim=1) # 返回最大值索引
        out = torch.cat([tmp_ar[:,idx[0], idx[1]].unsqueeze(0) for (tmp_ar, idx) in zip(outs, idxs_ij)], dim=0) # (b,2)求索引对应的logit

        # out = F.adaptive_max_pool2d(outs, (1, 1))
        out = out.view(out.size(0), -1) # (b,1)
        # print('out.shape', out.shape)
        return out
    
    def classifier(self, fea_map):
        x = self.relu(fea_map)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, x, return_feature=False):
        fea_map = self.features(x)
        fea_map = self.relu(fea_map)
        logits_cls = self.Sep_classifier(fea_map)
        # print(logits_cls)
        # logits_cls = logits_cls.squeeze(-1)
        if return_feature:
            # fea = self.relu(fea_map)
            feature = F.adaptive_avg_pool2d(fea_map, (1, 1))
            feature = feature.view(feature.size(0), -1)
            return logits_cls, feature
        else:
            return logits_cls

'''XceptionNet'''
pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975  # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}

# PRETAINED_WEIGHT_PATH = 'networks/xception-b5690688.pth'
PRETAINED_WEIGHT_PATH = './networks/xception-b5690688.pth'

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


def add_gaussian_noise(ins, mean=0, stddev=0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return ins + noise


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000, inc=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------
    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    def fea_part4(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        return x

    def fea_part5(self, x):
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)
        return x

    def classifier(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, input):
        x = self.features(input)
        out, x = self.classifier(x)
        return out, x


def xception(num_classes=1000, pretrained='imagenet', inc=3):
    model = Xception(num_classes=num_classes, inc=inc)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(
                settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """

    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0,
                 weight_norm=False, return_fea=False, inc=3):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.return_fea = return_fea

        if modelchoice == 'xception':

            def return_pytorch04_xception(pretrained=True):
                # Raises warning "src not broadcastable to dst" but thats fine
                model = xception(pretrained=False)
                if pretrained:
                    # Load model in torch 0.4+
                    model.fc = model.last_linear
                    del model.last_linear
                    state_dict = torch.load(
                        PRETAINED_WEIGHT_PATH)
                    for name, weights in state_dict.items():
                        if 'pointwise' in name:
                            state_dict[name] = weights.unsqueeze(
                                -1).unsqueeze(-1)
                    model.load_state_dict(state_dict)
                    model.last_linear = model.fc
                    del model.fc
                return model

            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                if weight_norm:
                    # print('Using Weight_Norm')
                    self.model.last_linear = nn.utils.weight_norm(
                        nn.Linear(num_ftrs, num_out_classes), name='weight')
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                # print('Using dropout', dropout)
                if weight_norm:
                    # print('Using Weight_Norm')
                    self.model.last_linear = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.utils.weight_norm(
                            nn.Linear(num_ftrs, num_out_classes), name='weight')
                    )

                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )

            if inc != 3:
                self.model.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
                nn.init.xavier_normal(self.model.conv1.weight.data, gain=0.02)

        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean=False, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on lib, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise NotImplementedError('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        out, x = self.model(x)
        if self.return_fea:
            return out, x
        else:
            return out

    def features(self, x):
        x = self.model.features(x)
        return x

    def classifier(self, x):
        out, x = self.model.classifier(x)
        return out, x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
            True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
            224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)

