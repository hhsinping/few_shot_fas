import sys

sys.path.append("../../")

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from fas import fas_model_fix, fas_model_adapt
import random
import numpy as np
from config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca
from datetime import datetime
import time
from timeit import default_timer as timer
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

device = "cuda"


def test(config):
  """
    Eval the final FAS model on the target data.
    python test.py --config [C/M/I/O/cefa/surf/wmca]
    Checkpoints ({casia/msu/replay/oulu/cefa/surf/wmca}_checkpoint.pth.tar) are
    loaded automatically
    """
  # load data
  src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real, src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real, src5_train_dataloader_fake, src5_train_dataloader_real, test_dataloader = get_dataset(
      config.src1_data, config.src1_train_num_frames, config.src2_data,
      config.src2_train_num_frames, config.src3_data,
      config.src3_train_num_frames, config.src4_data,
      config.src4_train_num_frames, config.src5_data,
      config.src5_train_num_frames, config.tgt_data, config.tgt_test_num_frames)

  net = fas_model_adapt(config.gamma, config.beta).to(device)
  net_ = torch.load(config.tgt_data + "_checkpoint.pth.tar")
  net.load_state_dict(net_["state_dict"])

  valid_args = eval(test_dataloader, net, True)
  best_model_HTER = valid_args[3]
  best_model_AUC = valid_args[4]

  print("HTER = ", best_model_HTER)
  print("AUC = ", best_model_AUC)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)
  args = parser.parse_args()

  if args.config == "I":
    config = configI
  if args.config == "C":
    config = configC
  if args.config == "M":
    config = configM
  if args.config == "O":
    config = configO
  if args.config == "cefa":
    config = config_cefa
  if args.config == "surf":
    config = config_surf
  if args.config == "wmca":
    config = config_wmca

  for attr in dir(config):
    if attr.find("__") == -1:
      print("%s = %r" % (attr, getattr(config, attr)))
  test(config)

