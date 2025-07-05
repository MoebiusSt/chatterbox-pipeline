import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import yaml
from tqdm import tqdm

try:
    from ..utils.metrics import MetricsCalculator
    from ..utils.audio_utils import save_audio
except ImportError:
    # Fallback for direct script execution
    from utils.metrics import MetricsCalculator
    from utils.audio_utils import save_audio

class Trainer:
    """Training class for CleanUNet model"""
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device,
                 checkpoint_dir: str = "models/checkpoints",
                 log_dir: str = "outputs/training_logs"):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_pesq = 0.0
        self.epochs_without_improvement = 0
        
        # Scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_mixed_precision = config.get('hardware', {}).get('mixed_precision', False)
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            sample_rate=config.get('model', {}).get('sample_rate', 24000)
        )
        
        print(f"Trainer initialized with device: {device}")
        print(f"Mixed precision: {self.use_mixed_precision}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_config = self.config.get('training', {})
        scheduler_type = scheduler_config.get('scheduler', 'ReduceLROnPlateau')
        
        if scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('scheduler_factor', 0.5),
                patience=scheduler_config.get('scheduler_patience', 10),
                verbose=True
            )
        elif scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('scheduler_step_size', 30),
                gamma=scheduler_config.get('scheduler_gamma', 0.1)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('num_epochs', 100)
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'l1_loss': 0.0, 'stft_loss': 0.0, 'high_band_stft_loss': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            clean = batch['clean'].to(self.device)
            noisy = batch['noisy'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    enhanced = self.model(noisy)
                    loss_dict = self.criterion(enhanced, clean)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('training', {}).get('gradient_clip_norm'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                enhanced = self.model(noisy)
                loss_dict = self.criterion(enhanced, clean)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('training', {}).get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                
                self.optimizer.step()
            
            # Update loss tracking
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key != 'total_loss':
                    loss_components[key] += value.item()
            
            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            self.global_step += 1
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return {'avg_loss': avg_loss, **loss_components}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'l1_loss': 0.0, 'stft_loss': 0.0, 'high_band_stft_loss': 0.0}
        
        # Reset metrics calculator
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(pbar):
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                
                # Forward pass
                enhanced = self.model(noisy)
                loss_dict = self.criterion(enhanced, clean)
                loss = loss_dict['total_loss']
                
                # Update loss tracking
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        loss_components[key] += value.item()
                
                # Update metrics
                for i in range(clean.shape[0]):
                    self.metrics_calculator.update(clean[i], enhanced[i])
                
                pbar.set_postfix({'val_loss': loss.item()})
        
        # Calculate average losses
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        # Get metrics
        metrics = self.metrics_calculator.compute_average()
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, self.current_epoch)
        for key, value in loss_components.items():
            self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)
        
        for key, value in metrics.items():
            if not key.endswith('_std'):
                self.writer.add_scalar(f'Val/{key.upper()}', value, self.current_epoch)
        
        return {'avg_loss': avg_loss, **loss_components, **metrics}
    
    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_pesq': self.best_val_pesq,
            'config': self.config
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            
            # Also save to final models directory
            final_dir = Path("models/final")
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / "cleanunet_best.pth"
            torch.save(checkpoint, final_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_pesq = checkpoint.get('best_val_pesq', 0.0)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('training', {}).get('num_epochs', 100)
        validation_freq = self.config.get('training', {}).get('validation_freq', 1)
        save_freq = self.config.get('training', {}).get('save_freq', 5)
        early_stopping_patience = self.config.get('training', {}).get('early_stopping_patience', 20)
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Validation frequency: {validation_freq}")
        print(f"Save frequency: {save_freq}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % validation_freq == 0:
                val_metrics = self.validate()
                
                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['avg_loss'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                is_best = False
                if val_metrics['pesq'] > self.best_val_pesq:
                    self.best_val_pesq = val_metrics['pesq']
                    self.best_val_loss = val_metrics['avg_loss']
                    is_best = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Save checkpoint
                if epoch % save_freq == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
                
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Print progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['avg_loss']:.4f} - "
                      f"Val Loss: {val_metrics['avg_loss']:.4f} - "
                      f"Val PESQ: {val_metrics['pesq']:.3f} - "
                      f"Time: {epoch_time:.1f}s")
            else:
                # Print training progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch + 1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['avg_loss']:.4f} - "
                      f"Time: {epoch_time:.1f}s")
        
        # Final checkpoint
        self.save_checkpoint(filename="final_checkpoint.pth")
        
        # Close tensorboard writer
        self.writer.close()
        
        print("Training completed!")
        print(f"Best validation PESQ: {self.best_val_pesq:.3f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_sample_outputs(self, output_dir: str = "outputs/samples"):
        """Save sample enhanced audio outputs"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        sample_rate = self.config.get('model', {}).get('sample_rate', 24000)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= 5:  # Only save first 5 samples
                    break
                
                clean = batch['clean'].to(self.device)
                noisy = batch['noisy'].to(self.device)
                filenames = batch['filenames']
                
                enhanced = self.model(noisy)
                
                # Save audio files
                for i in range(clean.shape[0]):
                    filename = filenames[i]
                    
                    # Save clean, noisy, and enhanced audio
                    save_audio(clean[i].cpu(), 
                              str(output_dir / f"{filename}_clean.wav"), 
                              sample_rate)
                    save_audio(noisy[i].cpu(), 
                              str(output_dir / f"{filename}_noisy.wav"), 
                              sample_rate)
                    save_audio(enhanced[i].cpu(), 
                              str(output_dir / f"{filename}_enhanced.wav"), 
                              sample_rate)
        
        print(f"Sample outputs saved to {output_dir}")

def create_trainer(model, criterion, optimizer, train_loader, val_loader, config, device):
    """Factory function to create trainer"""
    return Trainer(model, criterion, optimizer, train_loader, val_loader, config, device) 