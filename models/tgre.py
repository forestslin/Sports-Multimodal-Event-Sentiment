import torch
import torch.nn as nn
import torch.nn.functional as F

class TGRE(nn.Module):
    def __init__(self, num_modalities, feature_dim, hidden_dim, num_zones=5):
        super(TGRE, self).__init__()
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        
        # Intrinsic modality embeddings (r_i)
        self.modality_embeddings = nn.Parameter(torch.randn(num_modalities, hidden_dim))
        
        # Message passing MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention projection
        self.attn_proj = nn.Linear(hidden_dim * 2, 1)
        
        # GRU for sequential latent dynamics
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Semantic Zone-Based Embedding
        self.num_zones = num_zones
        self.centroids = nn.Parameter(torch.randn(num_zones, 2)) # c_k in R^2 (arousal, valence)
        self.semantic_proj = nn.Linear(hidden_dim, 2) # Project h to 2D
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, X_t, s_prev):
        # X_t: (batch_size, num_modalities, hidden_dim)
        # s_prev: (batch_size, num_modalities, hidden_dim)
        batch_size = X_t.size(0)
        
        h_t = X_t 
        messages = torch.zeros(batch_size, self.num_modalities, self.hidden_dim, device=X_t.device)
        
        # Cross-Modal Interaction Encoding
        for i in range(self.num_modalities):
            a_i = torch.zeros(batch_size, self.hidden_dim, device=X_t.device)
            attn_weights = []
            msg_list = []
            
            for j in range(self.num_modalities):
                if i == j:
                    attn_weights.append(torch.full((batch_size, 1), -1e9, device=X_t.device))
                    msg_list.append(torch.zeros(batch_size, self.hidden_dim, device=X_t.device))
                    continue
                
                r_i = self.modality_embeddings[i].expand(batch_size, -1)
                r_j = self.modality_embeddings[j].expand(batch_size, -1)
                
                # Directed message m_{j \to i}^t
                msg_input = torch.cat([h_t[:, j, :] - h_t[:, i, :], r_j, r_i], dim=-1)
                m_ji = self.msg_mlp(msg_input)
                msg_list.append(m_ji)
                
                # Attention \alpha_{ji}^t
                attn_input = torch.cat([h_t[:, j, :], h_t[:, i, :]], dim=-1)
                e_ji = F.leaky_relu(self.attn_proj(attn_input))
                attn_weights.append(e_ji)
                
            attn_weights = torch.cat(attn_weights, dim=-1) # (batch_size, num_modalities)
            alpha = F.softmax(attn_weights, dim=-1)
            
            for j in range(self.num_modalities):
                if i != j:
                    a_i += alpha[:, j].unsqueeze(1) * msg_list[j]
            
            # Residual block absorption
            a_i = a_i + h_t[:, i, :]
            messages[:, i, :] = a_i
            
        # Sequential Latent Dynamics
        s_t = torch.zeros_like(s_prev)
        for i in range(self.num_modalities):
            s_t[:, i, :] = self.gru(messages[:, i, :], s_prev[:, i, :])
            
        # Semantic Zone-Based Embedding
        p_t = self.semantic_proj(s_t) # (batch_size, num_modalities, 2)
        
        u_t = torch.zeros(batch_size, self.num_modalities, 2, device=X_t.device)
        for i in range(self.num_modalities):
            dist = torch.cdist(p_t[:, i, :], self.centroids) # (batch_size, num_zones)
            w_ik = F.softmax(-dist / self.temperature, dim=-1)
            u_t[:, i, :] = torch.matmul(w_ik, self.centroids)
            
        return s_t, u_t, p_t
