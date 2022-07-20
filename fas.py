import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
from torch.autograd import Variable
import random
import os
from torch.nn.utils.weight_norm import WeightNorm
import sys

sys.path.append('../../')
import third_party
import timm
from functools import partial


def l2_norm(input, axis=1):
  norm = torch.norm(input, 2, axis, True)
  output = torch.div(input, norm)
  return output


class feature_generator_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(feature_generator_adapt, self).__init__()
    self.vit = third_party.create_model(
        'vit_base_patch16_224', pretrained=True, gamma=gamma, beta=beta)

  def forward(self, input):
    feat, total_loss = self.vit.forward_features(input)
    return feat, total_loss


class feature_generator_fix(nn.Module):

  def __init__(self):
    super(feature_generator_fix, self).__init__()
    self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)

  def forward(self, input):
    feat = self.vit.forward_features(input).detach()
    return feat


class feature_embedder(nn.Module):

  def __init__(self):
    super(feature_embedder, self).__init__()
    self.bottleneck_layer_fc = nn.Linear(768, 512)
    self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
    self.bottleneck_layer_fc.bias.data.fill_(0.1)
    self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                          nn.Dropout(0.5))

  def forward(self, input, norm_flag=True):
    feature = self.bottleneck_layer(input)
    if (norm_flag):
      feature_norm = torch.norm(
          feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * (2)**0.5
      feature = torch.div(feature, feature_norm)
    return feature


class classifier(nn.Module):

  def __init__(self):
    super(classifier, self).__init__()
    self.classifier_layer = nn.Linear(512, 2)
    self.classifier_layer.weight.data.normal_(0, 0.01)
    self.classifier_layer.bias.data.fill_(0.0)

  def forward(self, input, norm_flag=True):
    if (norm_flag):
      self.classifier_layer.weight.data = l2_norm(
          self.classifier_layer.weight, axis=0)
      classifier_out = self.classifier_layer(input)
    else:
      classifier_out = self.classifier_layer(input)
    return classifier_out


class fas_model_adapt(nn.Module):

  def __init__(self, gamma, beta):
    super(fas_model_adapt, self).__init__()
    self.backbone = feature_generator_adapt(gamma, beta)
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature, total_loss = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature, total_loss


class fas_model_fix(nn.Module):

  def __init__(self):
    super(fas_model_fix, self).__init__()
    self.backbone = feature_generator_fix()
    self.embedder = feature_embedder()
    self.classifier = classifier()

  def forward(self, input, norm_flag=True):
    feature = self.backbone(input)
    feature = self.embedder(feature, norm_flag)
    classifier_out = self.classifier(feature, norm_flag)
    return classifier_out, feature

