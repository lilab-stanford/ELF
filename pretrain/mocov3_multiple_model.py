import torch
import torch.nn as nn
import torch.nn.functional as F
from .abmil import BatchedABMIL
from einops import rearrange 


class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024,dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             nn.Dropout(dropout) if dropout else nn.Identity(),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x) 

class Projection(nn.Module):
    def __init__(self, dim, embed_dim=1024,dropout=0.25):
        super(Projection, self).__init__()

        self.head = nn.Linear(dim, embed_dim)
        

    def forward(self, x):
        return self.head(x)
    

class ABMILEmbedding(nn.Module):
    def __init__(self, c_dim, embed_dim, num_heads=8, nr_layers=2, dropout=0.25):
        super().__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.c_dim = c_dim
        # use transformer instead of mamba
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=4*embed_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=nr_layers
        )
        
        self.embed = nn.ModuleDict({"768":Embed(768,768),
                                   "1024":Embed(1024,768),
                                   "1280":Embed(1280,768),
                                    "1536":Embed(1536,768)})
        
        self.proj = nn.ModuleDict({"768":Projection(768,768),
                                   "1024":Projection(1024,768),
                                   "1280":Projection(1280,768),
                                    "1536":Projection(1536,768)})
        
        self.num_heads = num_heads
        self.attn = nn.ModuleList([
            BatchedABMIL(
                input_dim=int(c_dim/num_heads),
                hidden_dim=int(c_dim/num_heads),
                dropout=dropout,
                n_classes=1
            ) for _ in range(self.num_heads)
        ])

    def forward(self, x, lens=None, infer=False):
        raw_x = x
        
        if lens is not None:
            assert len(x) == len(lens)
            all_resized_x = []
            for i in range(len(x)):
                sample_x = x[i, :, :lens[i].item()].unsqueeze(0)
                # The following step is absolutely critical—please do not change it. 
                # Using a learnable MLP for projection will severely degrade performance and ruin pretraining.
                # I’ve tested this extensively. An alternative is random projection rather than a learned one 
                # (see Figure 5 in https://arxiv.org/pdf/2104.02057), The results are not hugely different. You can try it.
                # Trust me on this—don’t waste your time using learnable projection. 
                # I only got this after spending over one week on an 8 NVIDIA H100 GPUs cluster, so sad.
                resized_x = F.interpolate(sample_x, size=768, mode='linear', align_corners=True) 
                resized_x = resized_x.squeeze(0)
                all_resized_x.append(resized_x)
            x = torch.stack(all_resized_x, dim=0)
        else:
            x = x
        
        x = self.norm(x)
        
        if self.num_heads > 1:
            h_ = rearrange(x, 'b t (e c) -> b t e c',c=self.num_heads)
            attention = []
            for i, attn_net in enumerate(self.attn):
                _, processed_attention = attn_net(h_[:, :, :, i], return_raw_attention = True) 
                attention.append(processed_attention)
            A = torch.stack(attention, dim=-1)
            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1)
            A = torch.transpose(A,2,1)
            A = F.softmax(A, dim=-1) 
        else: 
            A = self.attn[0](x)
            A = F.softmax(A, dim=-1)
                    
        feats = []
        raw_feats = []
        for i in range(len(lens)):
            dim_key = str(lens[i].item())
            A_i = A[i].unsqueeze(0)  # [1, 1, seq_len]
            raw_x_i = raw_x[i,:,:lens[i].item()].unsqueeze(0)
            h_i = torch.bmm(A_i, raw_x_i).squeeze(1)  # [1, feature_dim]
            feat = torch.bmm(A_i, x).squeeze(1)
            feats.append(feat)
            raw_feats.append(h_i)
        feats = torch.stack(feats, dim=0).squeeze(1)
        raw_feats = torch.stack(raw_feats, dim=0).squeeze(1)

        assert len(feats.shape)==2, feats.shape
        if infer:
            return raw_feats, feats, A
        else:
            return feats

class MoCo(nn.Module):
    def __init__(self, c_dim, embed_dim, num_heads=8, nr_mamba_layers=2, T=0.2, dropout=0.25):
        super().__init__()
        self.T = T
        self.base_enc = ABMILEmbedding(c_dim, embed_dim, num_heads, nr_layers=nr_mamba_layers, dropout=dropout)
        self.momentum_enc = ABMILEmbedding(c_dim, embed_dim, num_heads, nr_layers=nr_mamba_layers, dropout=0.0)
        
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2*embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2*embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        
        self.classifier1 = nn.Linear(embed_dim, 2)
        self.classifier2 = nn.Linear(embed_dim, 20)
        
        # initialize momentum encoder
        for param_b, param_m in zip(self.base_enc.parameters(), self.momentum_enc.parameters()):
            param_m.data.copy_(param_b.data)  
            param_m.requires_grad = False
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m=0.99):
        for param_b, param_m in zip(self.base_enc.parameters(), self.momentum_enc.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
    
    def forward(self, x1, x2, lens1, lens2, m=0.99, cancer=None, site=None, return_logits=False, infer=False):
        # ensure the input is on the correct device
        if cancer is not None:
            cancer = cancer.to(x1.device)
            site = site.to(x1.device)
        
        x1_enc = self.base_enc(x1, lens1, infer)
        x2_enc = self.base_enc(x2, lens2, infer)
        q1 = self.predictor(x1_enc)
        q2 = self.predictor(x2_enc)     
        
        p1 = self.classifier1(x1_enc)
        p2 = self.classifier1(x2_enc)
        
        s1 = self.classifier2(x1_enc)
        s2 = self.classifier2(x2_enc)
        
        with torch.no_grad():
            self._update_momentum_encoder(m=m)
            k1 = self.momentum_enc(x1, lens1, infer)
            k2 = self.momentum_enc(x2, lens2, infer)
        
        cls_loss = (nn.CrossEntropyLoss()(p1, cancer) + nn.CrossEntropyLoss()(p2, cancer) + nn.CrossEntropyLoss()(s1, site) + nn.CrossEntropyLoss()(s2, site)) / 4
        cont_loss = (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)) / 2
        
        if return_logits:
            return cls_loss, (torch.softmax(p1, dim=1) + torch.softmax(p2, dim=1)) / 2
        else:
            return cont_loss, cls_loss
    
    def contrastive_loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = torch.arange(N, dtype=torch.long, device=q.device)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # check if in distributed environment
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
