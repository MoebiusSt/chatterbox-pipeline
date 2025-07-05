#!/usr/bin/env python3
"""
Train SimpleUNet for TTS artifact removal - stable training with limited data
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.simple_unet import SimpleUNet, SimpleUNetLoss
from data.dataset import TTSArtifactDataset
from utils.metrics import MetricsCalculator

def train_epoch(model, criterion, optimizer, train_loader, device, epoch, writer, global_step):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'l1_loss': 0.0, 'stft_loss': 0.0}
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    
    for batch_idx, batch in enumerate(pbar):
        clean = batch['clean'].to(device)
        noisy = batch['noisy'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        enhanced = model(noisy)
        loss_dict = criterion(enhanced, clean)
        loss = loss_dict['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update loss tracking
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key != 'total_loss':
                loss_components[key] += value.item()
        
        # Log to tensorboard
        if global_step[0] % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step[0])
            for key, value in loss_dict.items():
                writer.add_scalar(f'Train/{key}', value.item(), global_step[0])
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        global_step[0] += 1
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    for key in loss_components:
        loss_components[key] /= len(train_loader)
    
    return {'avg_loss': avg_loss, **loss_components}

def validate(model, criterion, val_loader, device, sample_rate=24000):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    loss_components = {'l1_loss': 0.0, 'stft_loss': 0.0}
    
    metrics_calculator = MetricsCalculator(sample_rate)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            clean = batch['clean'].to(device)
            noisy = batch['noisy'].to(device)
            
            # Forward pass
            enhanced = model(noisy)
            loss_dict = criterion(enhanced, clean)
            loss = loss_dict['total_loss']
            
            # Update loss tracking
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    loss_components[key] += value.item()
            
            # Update metrics
            for i in range(clean.shape[0]):
                metrics_calculator.update(clean[i], enhanced[i])
            
            pbar.set_postfix({'val_loss': loss.item()})
    
    # Calculate average losses
    avg_loss = total_loss / len(val_loader)
    for key in loss_components:
        loss_components[key] /= len(val_loader)
    
    # Get metrics
    metrics = metrics_calculator.compute_average()
    
    return {'avg_loss': avg_loss, **loss_components, **metrics}

def main():
    parser = argparse.ArgumentParser(description='Train SimpleUNet for TTS artifact removal')
    parser.add_argument('--train_data', default='data/processed/train', help='Training data directory')
    parser.add_argument('--val_data', default='data/processed/validation', help='Validation data directory')
    parser.add_argument('--output_dir', default='outputs/training_simple', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'sample_rate': 24000,
        'n_fft': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'crop_length': 48000,
        'l1_weight': 0.8,
        'stft_weight': 0.2
    }
    
    # Create datasets
    train_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(args.train_data, 'clean'),
        noisy_dir=os.path.join(args.train_data, 'noisy'),
        config=model_config,
        mode='train'
    )
    
    val_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(args.val_data, 'clean'),
        noisy_dir=os.path.join(args.val_data, 'noisy'),
        config=model_config,
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = SimpleUNet(model_config).to(device)
    criterion = SimpleUNetLoss(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer with strong regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,  # Strong regularization
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = [0]  # Use list to make it mutable
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        train_metrics = train_epoch(
            model, criterion, optimizer, train_loader, device, epoch, writer, global_step
        )
        
        # Validation
        val_metrics = validate(model, criterion, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_metrics['avg_loss'])
        
        # Log to tensorboard
        writer.add_scalar('Val/Loss', val_metrics['avg_loss'], epoch)
        writer.add_scalar('Val/PESQ', val_metrics['pesq'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
        print(f"Val Loss: {val_metrics['avg_loss']:.4f}")
        print(f"Val PESQ: {val_metrics['pesq']:.3f}")
        
        # Save best model
        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']
            patience_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['avg_loss'],
                'val_pesq': val_metrics['pesq'],
                'model_config': model_config
            }, output_dir / 'best_model.pth')
            
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= 10:
            print("Early stopping triggered!")
            break
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Save training info
    training_info = {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'model_config': model_config,
        'total_parameters': total_params
    }
    
    with open(output_dir / 'training_info.yaml', 'w') as f:
        yaml.dump(training_info, f)
    
    writer.close()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main() 