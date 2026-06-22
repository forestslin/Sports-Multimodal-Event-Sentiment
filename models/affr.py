import torch
import torch.nn as nn
import torch.nn.functional as F

class AFFR(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(AFFR, self).__init__()
        
        # Raw flow field prediction
        self.flow_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # Flow in 2D sentiment space
        )
        
        # Sentiment Potential Field
        self.potential_mlp = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Scalar potential
        )
        
        # Terminal state projection
        self.classifier = nn.Linear(2, num_classes)
        
    def forward(self, s_t, p_t, phase_t, sigma=0.1):
        # s_t: (batch_size, num_modalities, hidden_dim)
        # p_t: (batch_size, num_modalities, 2)
        # phase_t: (batch_size, 1)
        batch_size = s_t.size(0)
        num_modalities = s_t.size(1)
        
        # Calculate raw flow F_{raw}^t
        F_raw = self.flow_mlp(s_t) # (batch_size, num_modalities, 2)
        
        F_tilde = torch.zeros_like(F_raw)
        
        for i in range(num_modalities):
            # Require grad to compute potential gradient w.r.t affective coordinates
            p_i = p_t[:, i, :].clone().detach().requires_grad_(True)
            
            # Evaluate potential field \Phi(p)
            phi_input = torch.cat([p_i, phase_t], dim=-1)
            phi = self.potential_mlp(phi_input) # (batch_size, 1)
            
            # \nabla\Phi(\hat{x}_i^t)
            grad_phi = torch.autograd.grad(
                outputs=phi, 
                inputs=p_i, 
                grad_outputs=torch.ones_like(phi),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Sentiment Potential Modulation
            F_tilde[:, i, :] = F_raw[:, i, :] + grad_phi
            
        # Beam-Based Trajectory Refinement (Stochastic Noise Injection)
        noise = torch.randn_like(F_tilde) * sigma
        F_refined = F_tilde + noise
        
        # Next state in sentiment space
        p_next = p_t + F_refined
        
        # Aggregate across modalities
        p_agg = p_next.mean(dim=1)
        
        # Final classification via Softmax output layer
        logits = self.classifier(p_agg)
        
        return logits, F_raw, p_t
