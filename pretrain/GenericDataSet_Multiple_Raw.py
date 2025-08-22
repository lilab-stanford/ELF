import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
from glob import glob
import pathlib
from tqdm import tqdm
import pandas as pd

class FeatDataset(Dataset):
    def __init__(self, base_path, csv_path, fm_list, num_feats=600, feat_len=1536, split='train', single_model=False):
        super().__init__()
        self.base_path = base_path
        self.fm_list = fm_list
        self.num_feats = num_feats
        self.feat_len = feat_len
        self.single_model = single_model
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.split = split
        print(f"Loaded {split} set with {len(self.df)} samples")
        self.csv_path = csv_path
        self.pat_list = list(self.df["slide_id"])
        self.cancer_list = list(self.df["cancer"])
        self.site_list = list(self.df["site"])
        self.h5_list = list(self.df["h5_path"])
        self.site_dict = {'Adrenal': 0, 'Bladder': 1, 'Brain': 2, 'Breast': 3, 'Cervix': 4, 'Colorectal': 5, 'Esophagus': 6, 'Heart': 7, 'Kidney': 8, 'Liver': 9, 'Lung': 10, 'Ovary': 11, 'Pancreas': 12, 'Prostate': 13, 'Skin': 14, 'Spleen': 15, 'Stomach': 16, 'Testis': 17, 'Thyroid': 18, 'Uterus': 19}
        print(self.fm_list)

    def __len__(self):
        return len(self.pat_list)

    def __getitem__(self, idx):
        pat = self.pat_list[idx]
        cancer = self.cancer_list[idx]
        site = self.site_list[idx]
        h5_path = self.h5_list[idx]
        site_idx = self.site_dict[site]
        h5_path_patching = os.path.join(self.base_path.replace("features", "patching"), '{}.h5'.format(h5_path.replace("h5_files", "patches")))

        # selected fms randomly
        if self.single_model:
            random_fm = np.random.choice(self.fm_list, size=2, replace=False)
            random_fm1 = random_fm[0]
            random_fm2 = random_fm[1]
            
            # load feat
            h5_path1 = os.path.join(self.base_path, random_fm1, '{}.h5'.format(h5_path))
            h5_path2 = os.path.join(self.base_path, random_fm2, '{}.h5'.format(h5_path))
            
            with h5py.File(h5_path1, 'r') as f1, h5py.File(h5_path2, 'r') as f2:
                # feat 1
                if random_fm1 == "virchow2":
                    if f1["features"][:].shape[1] < 1280:
                        print(f"Warning: {pat} has less than 1280 features in {random_fm1}")
                    wsi_feat1 = torch.tensor(f1["features"][:])[:,:1280]
                else:
                    wsi_feat1 = torch.tensor(f1["features"][:])
                lens1 = wsi_feat1.shape[1]
                # feat 2
                if random_fm2 == "virchow2":
                    if f2["features"][:].shape[1] < 1280:
                        print(f"Warning: {pat} has less than 1280 features in {random_fm2}")
                    wsi_feat2 = torch.tensor(f2["features"][:])[:,:1280]
                else:
                    wsi_feat2 = torch.tensor(f2["features"][:])
                lens2 = wsi_feat2.shape[1]
                # overlap ratio
                overlap_ratio = np.random.uniform(0.4, 0.9)
                
                # use the same index to extract features from two models
                feat1_a, feat1_b, feat2_a, feat2_b = self.pad_or_sample_with_corresponding_overlap(
                    wsi_feat1, 
                    wsi_feat2,
                    n=self.num_feats,
                    overlap_ratio=overlap_ratio,
                    k1=self.feat_len,
                    k2=self.feat_len
                )   
                
                # Ensure all tensors have exactly the same size - this is redundant now
                # but we'll keep it as a safety check
                feat1_a = feat1_a[:self.num_feats, :self.feat_len]
                feat1_b = feat1_b[:self.num_feats, :self.feat_len]
                feat2_a = feat2_a[:self.num_feats, :self.feat_len]
                feat2_b = feat2_b[:self.num_feats, :self.feat_len]
            model_name_dict = {"virchow2":0, "gigapath":1, "uni":2, "conch_v1_5":3, "h0":4}
            return (feat1_a, feat1_b, feat2_a, feat2_b, lens1, lens2, cancer, site_idx)
        
    def pad_or_sample_with_corresponding_overlap(self, x1: torch.Tensor, x2: torch.Tensor, 
                                            n=1024, overlap_ratio=0.5, 
                                            k1=1536, k2=1536) -> tuple:
        """
        Sample two sets of features from each of two models with partial overlap,
        ensuring that the spatial locations correspond between models.
        
        Args:
            x1: feature tensor from model 1 of shape (N1, feat_dim1)
            x2: feature tensor from model 2 of shape (N2, feat_dim2)
            n: desired number of features in each set
            overlap_ratio: ratio of features that should overlap between sets (0.0-1.0)
            k1: desired feature dimension for model 1
            k2: desired feature dimension for model 2
            
        Returns:
            tuple of (feat1_a, feat1_b, feat2_a, feat2_b, indices) with corresponding spatial locations
        """
        # get the length of two feature sets
        length1 = x1.shape[0]
        length2 = x2.shape[0]
        
        # determine the number of common features (take the minimum of the two)
        common_length = min(length1, length2)
        
        # calculate the number of overlapping features
        overlap_size = int(n * overlap_ratio)
        overlap_size = max(0, min(overlap_size, n))
        
        # calculate the number of unique features in each set
        unique_size = n - overlap_size
        
        # ensure there are enough features for both sets
        total_needed = overlap_size + 2 * unique_size
        
        # if there are not enough features, adjust the size of overlap and unique parts
        if common_length < total_needed:
            # if there are not enough features, reduce the size of unique parts
            available_for_unique = common_length - overlap_size
            if available_for_unique < 0:
                overlap_size = common_length // 3
                unique_size = (common_length - overlap_size) // 2
            else:
                unique_size = available_for_unique // 2
                
            overlap_size = min(overlap_size, n - unique_size)
        
        # create a random permutation, for two models
        perm_idx = torch.randperm(common_length)
        
        # select the indices of overlapping features
        overlap_indices = perm_idx[:overlap_size]
        
        # select the indices of unique features in set A
        unique_indices_a = perm_idx[overlap_size:overlap_size + unique_size]
        
        # select the indices of unique features in set B
        unique_indices_b = perm_idx[overlap_size + unique_size:overlap_size + 2*unique_size]
        
        # if the unique features in set B are not enough, resample from the beginning (avoiding used indices)
        if len(unique_indices_b) < unique_size:
            more_needed = unique_size - len(unique_indices_b)
            # create a mask, mark the used indices
            used_mask = torch.zeros(common_length, dtype=torch.bool)
            used_mask[overlap_indices] = True
            used_mask[unique_indices_a] = True
            used_mask[unique_indices_b] = True
            
            # find the unused indices
            unused_indices = torch.nonzero(~used_mask).squeeze(-1)
            
            if len(unused_indices) >= more_needed:
                # if there are enough unused indices, select from them
                extra_indices = unused_indices[torch.randperm(len(unused_indices))[:more_needed]]
            else:
                # otherwise, allow some indices to be reused (prioritize unused ones)
                extra_indices = torch.cat([
                    unused_indices,
                    perm_idx[torch.randperm(len(perm_idx))[:more_needed - len(unused_indices)]]
                ])
            
            unique_indices_b = torch.cat([unique_indices_b, extra_indices])
        
        # combine the indices of overlapping and unique features
        indices_a = torch.cat([overlap_indices, unique_indices_a])
        indices_b = torch.cat([overlap_indices, unique_indices_b])
        
        # shuffle the indices, ensure the overlapping features are not all at the beginning
        indices_a = indices_a[torch.randperm(len(indices_a))]
        indices_b = indices_b[torch.randperm(len(indices_b))]
        
        # use the indices to extract features from two models
        # two views of model 1
        feat1_a = x1[indices_a]
        feat1_b = x1[indices_b]
        
        # two views of model 2
        feat2_a = x2[indices_a]
        feat2_b = x2[indices_b]
        
        # handle the feature dimension (pad or truncate)
        # feature of model 1
        feat1_dim = x1.shape[1]
        
        # ensure all feature sets have n features
        # if the number of features is not enough, need to repeat and pad
        if feat1_a.shape[0] < n:
            # calculate the number of times to repeat
            repeats_needed = (n + feat1_a.shape[0] - 1) // feat1_a.shape[0]  # 向上取整
            # create repeated features
            feat1_a_repeated = feat1_a.repeat(repeats_needed, 1)
            feat1_b_repeated = feat1_b.repeat(repeats_needed, 1)
            feat2_a_repeated = feat2_a.repeat(repeats_needed, 1)
            feat2_b_repeated = feat2_b.repeat(repeats_needed, 1)
            
            # take the required number of features
            feat1_a = feat1_a_repeated[:n]
            feat1_b = feat1_b_repeated[:n]
            feat2_a = feat2_a_repeated[:n]
            feat2_b = feat2_b_repeated[:n]
        
        # ensure the number of features is correct
        feat1_a = feat1_a[:n]
        feat1_b = feat1_b[:n]
        feat2_a = feat2_a[:n]
        feat2_b = feat2_b[:n]
        
        # handle the feature dimension
        if k1 - feat1_dim > 0:
            pad_size1 = k1 - feat1_dim
            feat1_a = torch.cat([feat1_a, torch.zeros(feat1_a.shape[0], pad_size1)], dim=1)
            feat1_b = torch.cat([feat1_b, torch.zeros(feat1_b.shape[0], pad_size1)], dim=1)
        else:
            feat1_a = feat1_a[:, :k1]
            feat1_b = feat1_b[:, :k1]
        
        # feature of model 2
        feat2_dim = x2.shape[1]
        if k2 - feat2_dim > 0:
            pad_size2 = k2 - feat2_dim
            feat2_a = torch.cat([feat2_a, torch.zeros(feat2_a.shape[0], pad_size2)], dim=1)
            feat2_b = torch.cat([feat2_b, torch.zeros(feat2_b.shape[0], pad_size2)], dim=1)
        else:
            feat2_a = feat2_a[:, :k2]
            feat2_b = feat2_b[:, :k2]
        
        # final size check, ensure consistency
        feat1_a = feat1_a[:n, :k1]
        feat1_b = feat1_b[:n, :k1]
        feat2_a = feat2_a[:n, :k2]
        feat2_b = feat2_b[:n, :k2]
        
        return feat1_a, feat1_b, feat2_a, feat2_b
       

    def pad_or_sample_multiple(self, feats: torch.Tensor, coords: torch.Tensor, n=1024, k=1536) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad or sample both features and coordinates consistently
        Args:
            feats: feature tensor of shape (N, feat_dim)
            coords: coordinate tensor of shape (N, 2)
            n: desired number of features
            k: desired feature dimension
        Returns:
            tuple of (padded/sampled features, padded/sampled coordinates)
        """
        length = feats.shape[0]
        # use the same random permutation index to ensure the features and coordinates are corresponding
        perm_idx = torch.randperm(length)
        feats = feats[perm_idx][:n]
        coords = coords[perm_idx][:n]
        
        if length < n:
            repeats = (n - length) // length
            tmp_feats = feats
            tmp_coords = coords
            for _ in range(repeats):
                perm_idx = torch.randperm(length)
                feats = torch.cat([feats, tmp_feats[perm_idx]])
                coords = torch.cat([coords, tmp_coords[perm_idx]])
            
            resample_size = (n - length) % length
            if resample_size > 0:
                perm_idx = torch.randperm(len(feats))[:resample_size]
                feats = torch.cat([feats, feats[perm_idx]])
                coords = torch.cat([coords, coords[perm_idx]])
        
        # only pad the feature dimension, keep the original dimension of coordinates
        feat_len = feats.shape[1]
        if k-feat_len > 0:
            pad_size = k-feat_len
            feats = torch.cat([feats, torch.zeros(n, pad_size)], dim=1)
        
        return feats, coords
    
    def pad_or_sample(self, x: torch.Tensor, n=1024, k=1536) -> torch.Tensor:
        length = x.shape[0]
        x = x[torch.randperm(len(x))][:n]
        if length < n:
            repeats = (n - length) // length
            tmp = x
            for _ in range(repeats):
                x = torch.cat([x,tmp[torch.randperm(length)]])
            resample_size = (n - length) % length
            if resample_size > 0:
                x = torch.cat([x,x[torch.randperm(len(x))][:resample_size]])
        feat_len = x.shape[1]
        if k-feat_len>0:
            pad_size = k-feat_len
            x = torch.cat([x,torch.zeros(n,pad_size)],dim=1)
        else:
            x = x[:,:k] 
        return x
    
    def pad_or_sample_global_local(self, x: torch.Tensor, global_n=2048, local_n=768, k=1536) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample global features and local features, ensuring local features are a subset of global features.
        
        Args:
            x: feature tensor of shape (N, feat_dim)
            global_n: desired number of global features
            local_n: desired number of local features (must be <= global_n)
            k: desired feature dimension
            
        Returns:
            tuple of (global features, local features)
        """
        length = x.shape[0]
        
        # Ensure local_n is not greater than global_n
        local_n = min(local_n, global_n)
        
        # First randomly shuffle the features
        perm_idx = torch.randperm(length)
        x = x[perm_idx]
        
        # Handle case where we don't have enough features
        if length < global_n:
            repeats = (global_n - length) // length
            tmp = x
            for _ in range(repeats):
                x = torch.cat([x, tmp[torch.randperm(length)]])
            resample_size = (global_n - length) % length
            if resample_size > 0:
                x = torch.cat([x, x[torch.randperm(length)][:resample_size]])
        
        # Take the first global_n features as global features
        global_feats = x[:global_n]
        
        # Randomly select local_n features from global features
        local_indices = torch.randperm(global_n)[:local_n]
        local_feats = global_feats[local_indices]
        
        # Handle feature dimension padding or truncation
        # feat_len = x.shape[1]
        # if k - feat_len > 0:
        #     pad_size = k - feat_len
        #     global_feats = torch.cat([global_feats, torch.zeros(global_n, pad_size)], dim=1)
        #     local_feats = torch.cat([local_feats, torch.zeros(local_n, pad_size)], dim=1)
        # else:
        #     global_feats = global_feats[:, :k]
        #     local_feats = local_feats[:, :k]
            
        return global_feats, local_feats
    
    def pad_or_sample_with_partial_overlap(self, x: torch.Tensor, n=1024, overlap_ratio=0.5, k=1536) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample two sets of features with partial overlap between them.
        
        Args:
            x: feature tensor of shape (N, feat_dim)
            n: desired number of features in each set
            overlap_ratio: ratio of features that should overlap between the two sets (0.0-1.0)
            k: desired feature dimension
            
        Returns:
            tuple of (features_set1, features_set2) with partial overlap
        """
        length = x.shape[0]
        
        # First randomly shuffle the features
        perm_idx = torch.randperm(length)
        x = x[perm_idx]
        
        # Handle case where we don't have enough features
        if length < n:
            repeats = (n - length) // length
            tmp = x
            for _ in range(repeats):
                x = torch.cat([x, tmp[torch.randperm(length)]])
            resample_size = (n - length) % length
            if resample_size > 0:
                x = torch.cat([x, x[torch.randperm(length)][:resample_size]])
            length = x.shape[0]  # Update length after augmentation
        
        # Calculate number of overlapping features
        overlap_size = int(n * overlap_ratio)
        # Ensure overlap_size is valid
        overlap_size = max(0, min(overlap_size, n))
        
        # Calculate number of unique features for each set
        unique_size = n - overlap_size
        
        # Ensure we have enough features for both sets
        total_needed = overlap_size + 2 * unique_size
        if length < total_needed:
            # If not enough features, reduce unique portions equally
            available_for_unique = length - overlap_size
            unique_size = available_for_unique // 2
            overlap_size = n - unique_size
        
        # Select features for set 1
        # First take the overlapping portion
        overlap_features = x[:overlap_size]
        
        # Then take unique features for set 1
        unique_features1 = x[overlap_size:overlap_size + unique_size]
        
        # And unique features for set 2
        unique_features2 = x[overlap_size + unique_size:overlap_size + 2*unique_size]
        
        # If we still need more unique features for set 2, wrap around
        if unique_features2.shape[0] < unique_size:
            more_needed = unique_size - unique_features2.shape[0]
            extra_features = x[:(overlap_size + unique_size)]  # Reuse from the beginning
            perm_extra = torch.randperm(extra_features.shape[0])
            unique_features2 = torch.cat([unique_features2, extra_features[perm_extra][:more_needed]])
        
        # Combine overlapping and unique features for each set
        # Shuffle to ensure overlapping features aren't all at the beginning
        features_set1 = torch.cat([overlap_features, unique_features1])
        perm1 = torch.randperm(features_set1.shape[0])
        features_set1 = features_set1[perm1]
        
        features_set2 = torch.cat([overlap_features, unique_features2])
        perm2 = torch.randperm(features_set2.shape[0])
        features_set2 = features_set2[perm2]
        
        # Handle feature dimension padding or truncation
        # feat_len = x.shape[1]
        # if k - feat_len > 0:
        #     pad_size = k - feat_len
        #     features_set1 = torch.cat([features_set1, torch.zeros(n, pad_size)], dim=1)
        #     features_set2 = torch.cat([features_set2, torch.zeros(n, pad_size)], dim=1)
        # else:
        #     features_set1 = features_set1[:, :k]
        #     features_set2 = features_set2[:, :k]
        
        return features_set1, features_set2
