import sys
sys.path.append('/path/to/So-DAS/executor1')

import torch
import yaml
import numpy as np
from model import make_model_and_optimizer

def save_checkpoint(model, ori, sub1, sub2):
    checkpoint_ori = torch.load(ori, map_location='cpu')
    checkpoint_sub1 = torch.load(sub1, map_location='cpu')
    checkpoint_sub2 = torch.load(sub2, map_location='cpu')
    for k in list(checkpoint_ori['state_dict'].keys()):
        if 'doa_est' in k:
            checkpoint_ori['state_dict'][k] = checkpoint_sub1['state_dict'][k.replace('doa_est.', '')]

        if 'MISO2' in k:
            checkpoint_ori['state_dict'][k] = checkpoint_sub2['state_dict'][k]

    torch.save(checkpoint_ori, './exp/_ckpt_epoch_r2e_0.ckpt')
    print('checkpoint save Done.')

if __name__ == "__main__":
    with open("./exp/r2r/conf.yml") as f:
        train_conf = yaml.safe_load(f)

    model, _ = make_model_and_optimizer(train_conf)
    ori = './exp/_ckpt_epoch_r2e_0.ckpt'
    sub1 = '/path/to/DOA-Estimator/checkpoint/model'
    sub2 = '/path/to/Separator/checkpoint/model'
    model = save_checkpoint(model, ori, sub1, sub2)
