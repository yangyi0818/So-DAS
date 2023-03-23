import os
import random
import soundfile as sf
import torch
from torch import nn
import yaml
import json
import argparse
import numpy as np
from tqdm import tqdm

from asteroid import torch_utils
from asteroid.utils import tensors_to_device
from model import make_model_and_optimizer

parser = argparse.ArgumentParser()
parser.add_argument("--normalize", type=int, required=True)
parser.add_argument("--test_dir", type=str, required=True, help="Test directory including the json files")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = torch.mean(wav_tensor, dim=-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)
  
def load_best_model(model, exp_dir):
    try:
        with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
    print( 'LOADING from ',best_model_path)
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model)
    model = model.eval()
    return model


def main(conf):
    azimuth_resolution = np.array([2.5,5,10,15])
    True_est_azimuth1, True_est_azimuth2, True_est_azimuth = np.zeros(4), np.zeros(4), np.zeros(4)
    azimuth_mean_loss1, azimuth_mean_loss2, azimuth_mean_loss = 0, 0, 0

    model, _ = make_model_and_optimizer(train_conf)
    model = load_best_model(model, conf['exp_dir'])
    testset = conf['test_dir']

    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device

    normalize = conf['normalize']
    
    dlist = os.listdir(testset)
    pbar = tqdm(range(len(dlist)))
    torch.no_grad().__enter__()
    for idx in pbar:
        test_wav = np.load(testset + dlist[idx])
        mix, sources = tensors_to_device([torch.from_numpy(test_wav['mix']), torch.from_numpy(test_wav['src'])], device=model_device)
        name, single_speaker, tgt_doa = tensors_to_device([str(test_wav['n']), test_wav['single_speaker'], torch.from_numpy(test_wav['doa'])], device=model_device)

        if sources.dim() == 3:
            sources = sources[:,0]

        if (normalize):
            m_std = mix.std(1, keepdim=True)
            mix = normalize_tensor_wav(mix, eps=1e-8, std=m_std)
            sources = normalize_tensor_wav(sources, eps=1e-8, std=m_std[[0]]) # [s n]

        est_doa = model(mix[None]) # [b 2]

        tgt_doa = tgt_doa[:,0]

        # accuracy
        tgt_azimuth = (torch.atan2(tgt_doa[1], tgt_doa[0]) / np.pi * 180).cpu().numpy()
        est_azimuth = (torch.atan2(est_doa[0,1], est_doa[0,0]) / np.pi *180).cpu().numpy()

        error_azimuth = np.abs(tgt_azimuth - est_azimuth)
        if (error_azimuth > 180):
            error_azimuth = 360 - error_azimuth
        True_est_azimuth += (error_azimuth <= azimuth_resolution)
        azimuth_mean_loss += error_azimuth

        pbar.set_description(" {} {} {} {}".format('%.1f'%(azimuth_mean_loss / (idx+1)), '%.1f'%(error_azimuth), '%.1f'%(tgt_azimuth), '%.1f'%(est_azimuth)))

    azimuth_mean_loss /= len(dlist)
    print('azimuth MAE in degree: ', '%.2f'%(azimuth_mean_loss))
    for i in range (len(azimuth_resolution)):
        print('Acc. on azimuth resolution ', azimuth_resolution[i], ' : ', '%.3f'%(True_est_azimuth[i]/len(dlist)))
        
if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    main(arg_dic)
