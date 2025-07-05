#!/usr/bin/env python3
"""
Training script for CleanUNet TTS artifact removal
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cleanunet import CleanUNet
from models.loss import CleanUNetLoss
from data.dataset import TTSArtifactDataset, CollateFn
from training.trainer import Trainer
from utils.audio_utils import load_audio

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CleanUNet for TTS artifact removal')
    
    # Configuration files
    parser.add_argument('--config', default='config/train_config.yaml', 
                       help='Training configuration file')
    parser.add_argument('--model_config', default='config/model_config.yaml', 
                       help='Model configuration file')
    
    # Training options
    parser.add_argument('--resume', help='Resume training from checkpoint')
    parser.add_argument('--pretrained', help='Load pretrained model weights')
    parser.add_argument('--output_dir', default='outputs/training', 
                       help='Output directory for logs and checkpoints')
    
    # Data options
    parser.add_argument('--train_data', help='Override training data directory')
    parser.add_argument('--val_data', help='Override validation data directory')
    
    # Hardware options
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    parser.add_argument('--mixed_precision', action='store_true', 
                       help='Use mixed precision training')
    
    # Debugging options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--dry_run', action='store_true', help='Run without training')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_device(args, config):
    """Setup compute device"""
    if args.device:
        device = torch.device(args.device)
    elif config.get('hardware', {}).get('device'):
        device = torch.device(config['hardware']['device'])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def create_datasets(train_config, model_config, args):
    """Create training and validation datasets"""
    # Override data paths if provided
    train_data_dir = args.train_data or train_config['training']['train_data_dir']
    val_data_dir = args.val_data or train_config['training']['val_data_dir']
    
    # Check if data directories exist
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_data_dir}")
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(f"Validation data directory not found: {val_data_dir}")
    
    # Create datasets
    train_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(train_data_dir, 'clean'),
        noisy_dir=os.path.join(train_data_dir, 'noisy'),
        config=model_config['model'],
        mode='train'
    )
    
    val_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(val_data_dir, 'clean'),
        noisy_dir=os.path.join(val_data_dir, 'noisy'),
        config=model_config['model'],
        mode='val'
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def create_data_loaders(train_dataset, val_dataset, config):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=CollateFn(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=CollateFn(),
        drop_last=False
    )
    
    return train_loader, val_loader

def create_model(model_config, device):
    """Create CleanUNet model"""
    model = CleanUNet(model_config['model'])
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    return model

def create_optimizer(model, config):
    """Create optimizer"""
    optimizer_config = config['training']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    return optimizer

def load_pretrained_weights(model, pretrained_path):
    """Load pretrained model weights"""
    print(f"Loading pretrained weights from: {pretrained_path}")
    
                checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("Pretrained weights loaded successfully")

def validate_configuration(train_config, model_config):
    """Validate training configuration"""
    required_keys = ['training', 'hardware']
    for key in required_keys:
        if key not in train_config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    required_training_keys = ['batch_size', 'num_epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in train_config['training']:
            raise ValueError(f"Missing required training parameter: {key}")
    
    if 'model' not in model_config:
        raise ValueError("Missing model configuration")
    
    print("Configuration validation passed")

def main():
    """Main training function"""
    args = parse_args()
    
    # Load configurations
    print("Loading configurations...")
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    
    # Override mixed precision if specified
    if args.mixed_precision:
        train_config['hardware']['mixed_precision'] = True
    
    # Validate configuration
    validate_configuration(train_config, model_config)
    
    # Setup device
    device = setup_device(args, train_config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    with open(output_dir / 'train_config.yaml', 'w') as f:
        yaml.dump(train_config, f)
    with open(output_dir / 'model_config.yaml', 'w') as f:
        yaml.dump(model_config, f)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(train_config, model_config, args)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, train_config)
    
    # Create model
    print("Creating model...")
    model = create_model(model_config, device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        load_pretrained_weights(model, args.pretrained)
    
    # Create loss function
    print("Creating loss function...")
    criterion = CleanUNetLoss(model_config['loss']).to(device)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(model, train_config)
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        log_dir=str(output_dir / 'logs')
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Debug mode - just validate setup
    if args.debug:
        print("Debug mode - validating setup...")
        
        # Test forward pass
        sample_batch = next(iter(train_loader))
        clean = sample_batch['clean'].to(device)
        noisy = sample_batch['noisy'].to(device)
        
        with torch.no_grad():
            output = model(noisy)
            loss_dict = criterion(output, clean)
        
        print(f"Sample batch shape: {clean.shape}")
        print(f"Model output shape: {output.shape}")
        print(f"Sample loss: {loss_dict['total_loss'].item():.4f}")
        
        # Print model summary
        print("\nModel architecture:")
        print(model)
        
        return
    
    # Dry run mode - setup only
    if args.dry_run:
        print("Dry run mode - setup completed successfully")
        return
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model info
    final_info = {
        'training_completed': True,
        'best_val_loss': trainer.best_val_loss,
        'best_val_pesq': trainer.best_val_pesq,
        'total_epochs': trainer.current_epoch + 1,
        'device': str(device),
        'model_config': model_config,
        'train_config': train_config
    }
    
    with open(output_dir / 'training_info.yaml', 'w') as f:
        yaml.dump(final_info, f)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main() 