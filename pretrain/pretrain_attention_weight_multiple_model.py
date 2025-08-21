import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import OrderedDict, defaultdict
from torch.utils.tensorboard import SummaryWriter
import logging
import threading
import pandas as pd

# import custom modules
from mocov3_multiple_model import MoCo
from GenericDataSet_Multiple_Raw import FeatDataset

def parse_args():
    parser = argparse.ArgumentParser('Attention Weight training - Distributed')
    # base parameters
    parser.add_argument('--exp-name', type=str, default='attention_weight_exp_dist')
    parser.add_argument('--output-dir', type=str, default='/mnt/radonc-li02/private/luoxd96/elf/checkpoints/attention_weight_trans')
    
    # data parameters
    parser.add_argument('--data-path', type=str, required=True, help='base path for features')
    parser.add_argument('--csv-path', type=str, required=True, help='path to csv file')
    parser.add_argument('--num-feats', type=int, default=1024, help='number of features per sample')
    parser.add_argument('--feat-len', type=int, default=1536, help='feature length')
    # parser.add_argument('--feature-models', nargs='+', default=["uni", "gigapath", "h0", "virchow2", "conch_v1_5", "phikon_v2"])
    parser.add_argument('--feature-models', nargs='+', default=["uni"])
    
    # model parameters
    parser.add_argument('--dim', type=int, default=768, help='base feature dimension')
    parser.add_argument('--l-dim', type=int, default=1024, help='contrastive learning dimension')
    parser.add_argument('--nr-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--nr-layers', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--moco-t', type=float, default=0.2)
    parser.add_argument('--moco-m', type=float, default=0.999)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--print-freq', type=int, default=10, 
                        help='print frequency (default: 10)')
    
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0, 1],
                        help='GPU ids to use (e.g., --gpu 0 1)')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, 
                       dest='local_rank', help='local rank for distributed training')
    
    # other parameters
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint, or "auto" to resume from latest checkpoint')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--val-freq', type=int, default=2)
    
    # cache parameters
    parser.add_argument('--train-cache-size', type=int, default=10000, help='training set cache size')
    parser.add_argument('--val-cache-size', type=int, default=2000, help='validation set cache size')
    
    # new parameters
    parser.add_argument('--bf16', action='store_true', help='use bfloat16 for model')
    parser.add_argument('--mixed-precision', action='store_true', help='use mixed precision')
    
    args = parser.parse_args()
    return args

def main():
    # Add memory management settings
    # torch.cuda.set_per_process_memory_fraction(0.95)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    args = parse_args()
    
    # create output directory
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Get the number of available GPUs
    ngpus_per_node = torch.cuda.device_count()
    
    # force distributed mode: check local_rank
    args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.local_rank == -1:
        raise ValueError("Distributed mode expected. Please set LOCAL_RANK.")
    
    # Ensure local_rank is valid
    if args.local_rank >= ngpus_per_node:
        raise ValueError(f"Local rank {args.local_rank} is invalid for {ngpus_per_node} GPUs")

    # Set device before initializing process group
    torch.cuda.set_device(args.local_rank)
    
    # distributed training initialization
    args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    args.rank = int(os.environ.get('RANK', args.rank))
    args.gpu = args.local_rank
    
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    
    print(f'Process group initialized: rank {args.rank}, local_rank {args.local_rank}, '
          f'world_size {args.world_size}, gpu {args.gpu}, ngpus_per_node {ngpus_per_node}')
    
    main_worker(args.local_rank, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    # set gpu to local_rank under DDP
    args.gpu = args.local_rank
    args.batch_size = int(args.batch_size / args.world_size)
    args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size)

    device = torch.device('cuda:{}'.format(args.local_rank))
    print(f"Using device: {device}")
    
    # create model and convert to SyncBatchNorm, then move to GPU and wrap with DDP
    print("Creating model")
    model = MoCo(
        c_dim=args.l_dim,
        embed_dim=args.dim,
        num_heads=args.nr_heads,
        nr_mamba_layers=args.nr_layers,
        T=args.moco_t,
        dropout=args.dropout
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[args.rank], find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    # Add BF16/Mixed precision handling
    if args.bf16:
        scaler = None
        amp_dtype = torch.bfloat16
    else:
        scaler = GradScaler()
        amp_dtype = torch.float16 if args.mixed_precision else torch.float32

    if args.resume:
        if args.resume.lower() == 'auto':
            latest_checkpoint = os.path.join(args.output_dir, 'checkpoint_latest.pth')
            if os.path.isfile(latest_checkpoint):
                args.resume = latest_checkpoint
            else:
                print("No checkpoint found for auto-resume")
                args.resume = ''
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            map_location = f'cuda:{args.local_rank}'
            try:
                torch.load(args.resume, map_location='cpu', weights_only=False)  # Validate file integrity
            except Exception as e:
                print(f"Checkpoint file appears to be corrupted: {e}")
                backup_file = args.resume + '.backup'
                if os.path.isfile(backup_file):
                    print(f"Trying to load backup checkpoint: {backup_file}")
                    args.resume = backup_file
                else:
                    raise e
            
            checkpoint = torch.load(args.resume, map_location=map_location, weights_only=False)
            state_dict = checkpoint['state_dict']
            if not all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            print(f"Successfully loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    cudnn.benchmark = True
    
    # create dataset and dataloader (distributed sampler)
    train_dataset = FeatDataset(
        base_path=args.data_path,
        csv_path=args.csv_path,
        fm_list=args.feature_models,
        num_feats=args.num_feats,
        feat_len=args.feat_len,
        split='train',
        single_model=True
    )
    
    val_dataset = FeatDataset(
        base_path=args.data_path,
        csv_path=args.csv_path,
        fm_list=args.feature_models,
        num_feats=args.num_feats,
        feat_len=args.feat_len,
        split='test',
        single_model=True
    )
    
    # use balanced sampler instead of normal distributed sampler
    train_sampler = BalancedDistributedSampler(
        train_dataset, 
        num_replicas=args.world_size,
        rank=args.rank,
        balance_keys=['cancer', 'site']
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    # Add this diagnostic print after creating the train_loader
    if args.rank == 0:
        print(f"Train loader has {len(train_loader)} batches with batch size {args.batch_size}")
        print(f"Total samples per epoch: {len(train_loader) * args.batch_size * args.world_size}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    # only record Tensorboard and log on rank 0
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
        log_file = os.path.join(args.output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    best_overall_auc = 0
    best_site_auc = 0
    best_balanced_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        # Calculate momentum for current epoch
        m = adjust_moco_momentum(epoch, args)
        
        if args.rank == 0:
            logging.info(f'Epoch {epoch}: learning rate set to {lr:.6f}, momentum set to {m:.6f}')
            if writer is not None:
                writer.add_scalar('Train/LearningRate', lr, epoch)
                writer.add_scalar('Train/Momentum', m, epoch)
        train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}", 
                          disable=(args.rank != 0))
        
        # Add GPU cache clearing function
        def clear_gpu_cache():
            torch.cuda.empty_cache()

        for i, (feats1_a, feats1_b, feats2_a, feats2_b, lens1, lens2, cancer, site) in enumerate(train_iter):
            feats1_a = feats1_a.cuda(args.gpu, non_blocking=True)
            feats1_b = feats1_b.cuda(args.gpu, non_blocking=True)
            feats2_a = feats2_a.cuda(args.gpu, non_blocking=True)
            feats2_b = feats2_b.cuda(args.gpu, non_blocking=True)
            cancer = cancer.cuda(args.gpu, non_blocking=True)
            site = site.cuda(args.gpu, non_blocking=True)
            lens1 = lens1.cuda(args.gpu, non_blocking=True)
            lens2 = lens2.cuda(args.gpu, non_blocking=True)
            # with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(args.mixed_precision or args.bf16)):
            #     cont_loss, cls_loss = model(feats1_a, feats1_b, feats2_a, feats2_b, lens1, lens2, m=m, cancer=cancer, site=site)
            #     loss = (cont_loss + cls_loss) / 2
            
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=(args.mixed_precision or args.bf16)):
                cont_loss1, cls_loss1 = model(feats1_a, feats1_b, lens1, lens1, m=m, cancer=cancer, site=site)
                cont_loss2, cls_loss2 = model(feats2_a, feats2_b, lens2, lens2, m=m, cancer=cancer, site=site)
                cont_loss3, cls_loss3 = model(feats1_a, feats2_a, lens1, lens2, m=m, cancer=cancer, site=site)
                cont_loss4, cls_loss4 = model(feats1_b, feats2_b, lens1, lens2, m=m, cancer=cancer, site=site)
                cont_loss = (1.5 * cont_loss1 + 1.5 * cont_loss2 + 0.5 * cont_loss3 + 0.5 * cont_loss4) / 4
                cls_loss = (cls_loss1 + cls_loss2 + cls_loss3 + cls_loss4) / 4
                loss = (cont_loss + cls_loss) / 2

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if i > 0:  # Clear cache periodically
                clear_gpu_cache()
            
            torch.cuda.synchronize()
            dist.barrier()
            
            optimizer.zero_grad()
            
            total_loss += loss.item()
            train_iter.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if i % args.print_freq == 0 and args.rank == 0:
                logging.info(
                    f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Cont Loss {cont_loss.item():.4f}\t'
                    f'Cls Loss {cls_loss.item():.4f}\t'
                    f'Total Loss {loss.item():.4f}'
                )
        
        if args.rank == 0:
            logging.info(f'Epoch {epoch} completed')
        
        if epoch > 0 and epoch % 5 == 0 and args.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_overall_auc': best_overall_auc,
                'best_site_auc': best_site_auc,
                'best_balanced_acc': best_balanced_acc,
                'optimizer': optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()
            
            save_checkpoint(checkpoint, is_best=False, output_dir=args.output_dir)
        
        if (epoch + 1) % args.val_freq == 0:
            # Store current epoch in args for logging purposes
            args.current_epoch = epoch
            args.writer = writer  # Make writer accessible in validate function
            
            val_loss, val_site_auc, val_balanced_acc = validate(val_loader, model, args)
            
            if args.rank == 0:
                logging.info(
                    f'Validation Epoch: [{epoch}]\t'
                    f'Val Loss {val_loss:.4f}\t'
                    f'Val Mean Site AUC {val_site_auc:.4f}\t'
                    f'Val Mean Balanced Acc {val_balanced_acc:.4f}'
                )
                
                # Use mean site AUC as the criterion for best model
                is_best = val_site_auc > best_site_auc
                best_site_auc = max(val_site_auc, best_site_auc)
                best_balanced_acc = max(val_balanced_acc, best_balanced_acc)
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_site_auc': best_site_auc,
                    'best_balanced_acc': best_balanced_acc,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                }, is_best, args.output_dir)
    
    if writer is not None:
        writer.close()

def validate(val_loader, model, args):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_sites = []
    
    with torch.no_grad():
        for feats1_a, feats1_b, feats2_a, feats2_b, lens1, lens2, cancer, site in val_loader:
            feats1_a = feats1_a.cuda(args.gpu, non_blocking=True)
            feats1_b = feats1_b.cuda(args.gpu, non_blocking=True)
            feats2_a = feats2_a.cuda(args.gpu, non_blocking=True)
            feats2_b = feats2_b.cuda(args.gpu, non_blocking=True)
            cancer = cancer.cuda(args.gpu, non_blocking=True)
            site = site.cuda(args.gpu, non_blocking=True)
            lens1 = lens1.cuda(args.gpu, non_blocking=True)
            lens2 = lens2.cuda(args.gpu, non_blocking=True)
            
            loss, logits = model(feats1_a, feats1_b, lens1, lens1, m=args.moco_m, cancer=cancer, site=site, return_logits=True)
            total_loss += loss.item()
            all_preds.extend(logits[:, 1].cpu().numpy())
            all_labels.extend(cancer.cpu().numpy())
            all_sites.extend(site.cpu().numpy())
    
    all_preds = torch.tensor(all_preds).cuda(args.gpu)
    all_labels = torch.tensor(all_labels).cuda(args.gpu)
    all_sites = torch.tensor(all_sites).cuda(args.gpu)
    
    # Gather predictions, labels, and sites
    gathered_preds = gather_tensors(all_preds)
    gathered_labels = gather_tensors(all_labels)
    gathered_sites = gather_tensors(all_sites)
    
    all_preds = gathered_preds.cpu().numpy()
    all_labels = gathered_labels.cpu().numpy()
    all_sites = gathered_sites.cpu().numpy()
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(all_labels, all_preds)
    
    # Calculate metrics per site
    site_metrics = {}
    unique_sites = np.unique(all_sites)
    
    for site_id in unique_sites:
        site_mask = (all_sites == site_id)
        site_preds = all_preds[site_mask]
        site_labels = all_labels[site_mask]
        
        # Check if we have both positive and negative samples
        if len(np.unique(site_labels)) < 2:
            if args.rank == 0:
                logging.info(f"Site {site_id}: Skipping - only one class present")
            continue
            
        # Convert predictions to binary using 0.5 threshold
        site_pred_binary = (site_preds > 0.5).astype(int)
        
        # Calculate sensitivity (true positive rate)
        pos_mask = (site_labels == 1)
        sensitivity = np.mean(site_pred_binary[pos_mask] == site_labels[pos_mask]) if np.any(pos_mask) else 0
        
        # Calculate specificity (true negative rate)
        neg_mask = (site_labels == 0)
        specificity = np.mean(site_pred_binary[neg_mask] == site_labels[neg_mask]) if np.any(neg_mask) else 0
        
        # Balanced accuracy is the average of sensitivity and specificity
        balanced_acc = (sensitivity + specificity) / 2
        
        # Calculate site-specific AUC
        site_auc = roc_auc_score(site_labels, site_preds)
        
        site_metrics[int(site_id)] = {
            'balanced_acc': balanced_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': site_auc,
            'samples': len(site_labels),
            'positive_samples': np.sum(site_labels),
            'negative_samples': len(site_labels) - np.sum(site_labels)
        }
        
        if args.rank == 0:
            logging.info(f"Site {site_id}: AUC = {site_auc:.4f}, Balanced Acc = {balanced_acc:.4f}, "
                         f"Sensitivity = {sensitivity:.4f}, Specificity = {specificity:.4f}, "
                         f"Samples = {len(site_labels)} (Pos: {np.sum(site_labels)}, Neg: {len(site_labels) - np.sum(site_labels)})")
    
    # Calculate mean metrics across all sites
    if site_metrics:
        mean_balanced_acc = np.mean([m['balanced_acc'] for m in site_metrics.values()])
        mean_site_auc = np.mean([m['auc'] for m in site_metrics.values()])
    else:
        mean_balanced_acc = 0
        mean_site_auc = 0
    
    if args.rank == 0:
        logging.info(f"Overall AUC: {overall_auc:.4f}")
        logging.info(f"Mean Site AUC: {mean_site_auc:.4f}")
        logging.info(f"Mean Balanced Accuracy: {mean_balanced_acc:.4f}")
        
        # Add these metrics to tensorboard if available
        if hasattr(args, 'writer') and args.writer is not None:
            args.writer.add_scalar('Val/OverallAUC', overall_auc, args.current_epoch)
            args.writer.add_scalar('Val/MeanSiteAUC', mean_site_auc, args.current_epoch)
            args.writer.add_scalar('Val/MeanBalancedAcc', mean_balanced_acc, args.current_epoch)
            
            # Log individual site metrics
            for site_id, metrics in sorted(site_metrics.items()):
                args.writer.add_scalar(f'Val/Site{site_id}/AUC', metrics['auc'], args.current_epoch)
                args.writer.add_scalar(f'Val/Site{site_id}/BalancedAcc', metrics['balanced_acc'], args.current_epoch)
                args.writer.add_scalar(f'Val/Site{site_id}/Sensitivity', metrics['sensitivity'], args.current_epoch)
                args.writer.add_scalar(f'Val/Site{site_id}/Specificity', metrics['specificity'], args.current_epoch)
    
    return total_loss / len(val_loader), mean_site_auc, mean_balanced_acc

@torch.no_grad()
def gather_tensors(tensor):
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    return torch.cat(output, dim=0)

def save_checkpoint(state, is_best, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    latest_filename = os.path.join(output_dir, 'checkpoint_latest.pth')
    best_filename = os.path.join(output_dir, 'checkpoint_best.pth')
    
    if is_best:
        temp_filename = best_filename + '.tmp'
        backup_filename = best_filename + '.backup'
    else:
        temp_filename = latest_filename + '.tmp'
        backup_filename = latest_filename + '.backup'
    
    try:
        torch.save(state, temp_filename)
        # Add weights_only=False to allow loading NumPy objects
        torch.load(temp_filename, map_location='cpu', weights_only=False)  # Validate
        os.replace(temp_filename, filename)
        torch.save(state, backup_filename)
        
        torch.save(state, latest_filename + '.tmp')
        os.replace(latest_filename + '.tmp', latest_filename)
        
        if is_best:
            torch.save(state, best_filename + '.tmp')
            os.replace(best_filename + '.tmp', best_filename)
            
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        def safe_remove(file_path):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {file_path}: {e}")
        
        safe_remove(temp_filename)
        safe_remove(latest_filename + '.tmp')
        safe_remove(best_filename + '.tmp')
        return

def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_moco_momentum(epoch, args):
    return 1. - 0.5 * (1. + np.cos(np.pi * epoch / args.epochs)) * (1. - args.moco_m)

class BalancedDistributedSampler(Sampler):
    """
    Sampler that ensures balanced sampling across specified keys (cancer, site)
    while maintaining compatibility with DistributedDataParallel.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, balance_keys=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package is not available")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.balance_keys = balance_keys or ['cancer']
        
        # Get the dataframe from the dataset
        self.df = pd.read_csv(dataset.csv_path)
        self.df = self.df[self.df['split'] == dataset.split]
        
        # Store the valid indices range
        self.valid_indices_range = len(dataset)
        
        # Create indices grouped by balance keys
        self.grouped_indices = self._create_grouped_indices()
        
        # Calculate total size ensuring divisibility by num_replicas
        self.total_size = self._calculate_total_size()
        self.num_samples = self.total_size // self.num_replicas
        
        # Print diagnostic information
        if rank == 0:
            print(f"BalancedDistributedSampler initialized with:")
            print(f"  - Dataset size: {len(dataset)}")
            print(f"  - Number of groups: {len(self.grouped_indices)}")
            print(f"  - Total size: {self.total_size}")
            print(f"  - Samples per rank: {self.num_samples}")
        
    def _create_grouped_indices(self):
        """Create indices grouped by the balance keys"""
        grouped_indices = defaultdict(list)
        
        # Group indices by the combination of balance keys
        valid_indices = list(range(min(len(self.df), self.valid_indices_range)))
        
        for idx in valid_indices:
            if idx < len(self.df):
                row = self.df.iloc[idx]
                key = tuple(row[k] for k in self.balance_keys)
                grouped_indices[key].append(idx)
            
        return grouped_indices
        
    def _calculate_total_size(self):
        """Calculate total size ensuring it's divisible by num_replicas"""
        # Find the minimum number of samples per group
        if not self.grouped_indices:
            return 0
            
        # Get the average number of samples per group
        avg_group_size = sum(len(indices) for indices in self.grouped_indices.values()) // len(self.grouped_indices)
        
        # Use a reasonable size that's not too small
        samples_per_group = max(16, avg_group_size)
        
        # Total size is samples_per_group multiplied by number of groups
        total_size = samples_per_group * len(self.grouped_indices)
        
        # Ensure total_size is divisible by num_replicas
        if total_size % self.num_replicas != 0:
            total_size = ((total_size // self.num_replicas) + 1) * self.num_replicas
            
        return total_size
        
    def __iter__(self):
        # Check if we have any valid indices
        if not self.grouped_indices or self.total_size == 0:
            print(f"Warning: No valid indices found for rank {self.rank}. Returning empty iterator.")
            return iter([])
            
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        
        # For each group, select indices
        for key, indices in self.grouped_indices.items():
            if self.shuffle:
                # Shuffle indices within each group
                indices = indices.copy()
                random.Random(self.seed + self.epoch).shuffle(indices)
            
            # Calculate how many samples to take from this group
            # We want to take roughly the same number from each group
            target_samples = self.total_size // len(self.grouped_indices)
            
            # If not enough samples in this group, repeat indices
            if len(indices) < target_samples:
                indices = indices * (target_samples // len(indices) + 1)
            
            # Take the required number of samples
            all_indices.extend(indices[:target_samples])
        
        # Shuffle all selected indices
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(all_indices)
        
        # Ensure total size is correct
        all_indices = all_indices[:self.total_size]
        
        # Verify all indices are within valid range
        all_indices = [idx for idx in all_indices if idx < self.valid_indices_range]
        
        # If we lost indices due to filtering, adjust the total size
        if len(all_indices) < self.total_size:
            # Repeat indices to reach the required total size
            deficit = self.total_size - len(all_indices)
            additional_indices = all_indices[:deficit]
            all_indices.extend(additional_indices)
            
        # Subsample for this specific rank
        indices_for_rank = all_indices[self.rank:self.total_size:self.num_replicas]
        
        # Print diagnostic information for the first epoch
        if self.epoch == 0 and self.rank == 0:
            print(f"Rank {self.rank}: Returning {len(indices_for_rank)} indices")
        
        return iter(indices_for_rank)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == '__main__':
    main() 