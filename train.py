import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultimodalSportsDataset
from models.framework import MultimodalAnticipatoryFramework
import argparse

def compute_divergence_and_curl(F_raw, p_t):
    # F_raw: (batch_size, num_modalities, 2)
    # p_t: (batch_size, num_modalities, 2)
    
    div_loss = 0.0
    curl_loss = 0.0
    batch_size, num_modalities, _ = F_raw.size()
    
    # Surrogate: Penalize sudden changes in F_raw across modalities (spatial divergence equivalent)
    # div ~ sum |F_i - F_j|
    for i in range(num_modalities):
        for j in range(i+1, num_modalities):
            diff = F_raw[:, i, :] - F_raw[:, j, :]
            div_loss += torch.norm(diff, p=1, dim=1).sum()
            
    # Curl surrogate: Cross product magnitude |F_i x F_j|
    for i in range(num_modalities):
        for j in range(i+1, num_modalities):
            # 2D cross product magnitude: |F_ix * F_jy - F_iy * F_jx|
            cross = F_raw[:, i, 0] * F_raw[:, j, 1] - F_raw[:, i, 1] * F_raw[:, j, 0]
            curl_loss += torch.abs(cross).sum()
            
    return div_loss / batch_size, curl_loss / batch_size

def train(epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    lr = 1e-4
    lambda_div = 0.1
    lambda_curl = 0.1
    
    # Dataset
    print("Loading Multimodal Mock Dataset...")
    dataset = MultimodalSportsDataset(num_samples=640, seq_len=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = MultimodalAnticipatoryFramework(
        num_modalities=3, 
        feature_dim=256, 
        hidden_dim=256, 
        num_zones=5, 
        num_classes=4
    ).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_div_loss = 0.0
        epoch_curl_loss = 0.0
        
        for X_seq, phase_seq, labels_seq in dataloader:
            X_seq = X_seq.to(device)
            phase_seq = phase_seq.to(device)
            labels_seq = labels_seq.to(device)
            
            optimizer.zero_grad()
            
            logits_seq, F_raw_seq, p_t_seq = model(X_seq, phase_seq)
            
            # Compute classification loss
            logits_flat = logits_seq.view(-1, 4)
            labels_flat = labels_seq.view(-1)
            ce_loss = criterion(logits_flat, labels_flat)
            
            # Compute physics-informed penalties
            div_loss = 0.0
            curl_loss = 0.0
            seq_len = X_seq.size(1)
            
            for t in range(seq_len):
                d_l, c_l = compute_divergence_and_curl(F_raw_seq[:, t, :, :], p_t_seq[:, t, :, :])
                div_loss += d_l
                curl_loss += c_l
                
            div_loss = div_loss / seq_len
            curl_loss = curl_loss / seq_len
            
            total_loss = ce_loss + lambda_div * div_loss + lambda_curl * curl_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_div_loss += div_loss.item()
            epoch_curl_loss += curl_loss.item()
            
        scheduler.step()
        
        num_batches = len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/num_batches:.4f} | CE: {epoch_ce_loss/num_batches:.4f} | Div: {epoch_div_loss/num_batches:.4f} | Curl: {epoch_curl_loss/num_batches:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train(epochs=args.epochs)
