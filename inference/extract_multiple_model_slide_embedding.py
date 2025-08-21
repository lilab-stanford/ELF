import torch
import glob
import os
import torch.nn as nn
import sys
import os
# Add the ELF directory to the Python path for direct script execution
elf_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, elf_dir)
from pretrain.mocov3_multiple_model import MoCo
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path


def parse_args():
    parser = argparse.ArgumentParser('COBRA Feature Extraction')
        # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to pretrained model checkpoint')
    parser.add_argument('--dim', type=int, default=768,
                        help='base feature dimension')
    parser.add_argument('--l-dim', type=int, default=768,
                        help='output feature dimension')
    parser.add_argument('--nr-heads', type=int, default=8,
                        help='number of attention heads')
    parser.add_argument('--nr-layers', type=int, default=6,
                        help='number of transformer layers')
    
    # data parameters
    parser.add_argument('--dataset', type=str, default="None",
                        help='base path for input features')
    parser.add_argument('--csv-path', type=str, default="None",
                        help='path to csv file')
    parser.add_argument('--output-path', type=str, default="None",
                        help='path to save extracted features')
    parser.add_argument('--num-feats', type=int, default=1024,
                        help='number of features per sample')
    parser.add_argument('--feat-len', type=int, default=1536,
                        help='feature length')
    # parser.add_argument('--feature-models', nargs='+', 
    #                     default=["uni", "gigapath", "h0", "virchow2", "conch_v1_5", "phikon_v2"])
    parser.add_argument('--feature-models', nargs='+', 
                        default=["conch_v1_5"])
    
    # running parameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()

def load_model(args, device):
    # create model
    model = MoCo(
        c_dim=args.l_dim,
        embed_dim=args.dim,
        num_heads=args.nr_heads,
        nr_mamba_layers=args.nr_layers,
        T=0.2,
        dropout=0.25
    )
    
    # load checkpoint, set weights_only=False to handle numpy arrays
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # handle the prefix of state dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # remove "module." prefix
    
    # load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model



def main_fp():
    args = parse_args()
    dataset = args.dataset
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # load model
    model = load_model(args, device)
    print("Model loaded successfully")

    fm_list = args.feature_models
    
    all_patients = os.listdir("/mnt/radonc-li02/private/luoxd96/omnipath/features/{}/{}/h5_files".format(fm_list[0], dataset))

    for patient in tqdm(all_patients):
        if os.path.exists("{}/{}".format(save_dir, patient)):
            print("{} already exists".format(patient))
            continue
        else:
            all_features = []
            all_lengths = []
            for fm in fm_list:
                h5py_file = h5py.File("/mnt/radonc-li02/private/luoxd96/omnipath/features/{}/{}/h5_files/{}".format(fm, dataset, patient), "r")
                if fm == "virchow2":
                    features = (h5py_file["features"][:][:, :1280] + h5py_file["features"][:][:, 1280:]) / 2
                else:
                    features = h5py_file["features"][:]
                all_features.append(features)
                all_lengths.append(features.shape[1])
            all_features = np.array(all_features)
            
            features = torch.from_numpy(all_features).float().to(device) 
            lengths = torch.tensor(all_lengths).to(device)
            
            with torch.no_grad():
                slide_embedding_raw, slide_embedding, attention_weights = model.momentum_enc(features, lens=lengths, infer=True)
                # slide_embedding_raw, slide_embedding, attention_weights = model.base_enc(features, lens=lengths, infer=True)
                slide_embedding_raw = slide_embedding_raw.detach().to(torch.float32).cpu().numpy()
                slide_embedding = slide_embedding.detach().to(torch.float32).cpu().numpy()
                attention_weights = attention_weights.detach().to(torch.float32).cpu().numpy()
                save_hdf5("{}/{}".format(save_dir, patient), {'features_dim': slide_embedding_raw, 'features': slide_embedding, 'attention_weights': attention_weights})
                print("{} saved".format(patient))
            
if __name__ == '__main__':
    main_fp()
