import torch
import torch.nn as nn
from .tgre import TGRE
from .affr import AFFR

class MultimodalAnticipatoryFramework(nn.Module):
    def __init__(self, num_modalities=3, feature_dim=256, hidden_dim=256, num_zones=5, num_classes=4):
        super(MultimodalAnticipatoryFramework, self).__init__()
        
        # Project heterogeneous features to uniform latent space
        self.feature_proj = nn.ModuleList([
            nn.Linear(feature_dim, hidden_dim) for _ in range(num_modalities)
        ])
        
        self.tgre = TGRE(num_modalities, hidden_dim, hidden_dim, num_zones)
        self.affr = AFFR(hidden_dim, num_classes)
        
    def forward(self, X_seq, phase_seq):
        # X_seq: (batch_size, seq_len, num_modalities, feature_dim)
        # phase_seq: (batch_size, seq_len, 1)
        batch_size, seq_len, num_modalities, feature_dim = X_seq.size()
        
        s_t = torch.zeros(batch_size, num_modalities, self.tgre.hidden_dim, device=X_seq.device)
        
        logits_seq = []
        F_raw_seq = []
        p_t_seq = []
        
        for t in range(seq_len):
            X_t = X_seq[:, t, :, :]
            
            # Modal feature projection
            h_t = torch.zeros_like(s_t)
            for i in range(num_modalities):
                h_t[:, i, :] = self.feature_proj[i](X_t[:, i, :])
                
            # TGRE encoding step
            s_t, u_t, p_t = self.tgre(h_t, s_t)
            
            # AFFR anticipation step
            logits, F_raw, p_t_mod = self.affr(s_t, p_t, phase_seq[:, t, :])
            
            logits_seq.append(logits.unsqueeze(1))
            F_raw_seq.append(F_raw.unsqueeze(1))
            p_t_seq.append(p_t_mod.unsqueeze(1))
            
        logits_seq = torch.cat(logits_seq, dim=1)
        F_raw_seq = torch.cat(F_raw_seq, dim=1)
        p_t_seq = torch.cat(p_t_seq, dim=1)
        
        return logits_seq, F_raw_seq, p_t_seq
