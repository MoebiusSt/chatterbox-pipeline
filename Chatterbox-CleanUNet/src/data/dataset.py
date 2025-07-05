import torch
from torch.utils.data import Dataset
import torchaudio
import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import soundfile as sf

class TTSArtifactDataset(Dataset):
    """Dataset for TTS artifact removal training"""
    
    def __init__(self, clean_dir: str, noisy_dir: str, config: Dict, mode: str = 'train'):
        """
        Initialize dataset
        
        Args:
            clean_dir: Path to clean audio files
            noisy_dir: Path to noisy audio files
            config: Configuration dictionary
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.config = config
        self.mode = mode
        self.sample_rate = config.get('sample_rate', 24000)
        self.crop_length = config.get('crop_length', 48000)
        
        # Augmentation settings
        self.use_augmentation = config.get('use_augmentation', False) and mode == 'train'
        self.time_stretch_range = config.get('time_stretch_range', [0.95, 1.05])
        self.pitch_shift_range = config.get('pitch_shift_range', [-1, 1])
        self.gain_range = config.get('gain_range', [0.0, 0.0])  # Disabled by default
        
        # Get matching clean/noisy pairs
        self.file_pairs = self._get_file_pairs()
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No matching file pairs found in {clean_dir} and {noisy_dir}")
        
        print(f"Loaded {len(self.file_pairs)} file pairs for {mode} mode")
        if self.use_augmentation:
            print(f"Augmentation enabled - Time stretch: {self.time_stretch_range}, Pitch shift: {self.pitch_shift_range}, Gain: {self.gain_range}")
        
    def _get_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Get matching clean/noisy file pairs"""
        clean_files = list(self.clean_dir.glob("*.wav"))
        pairs = []
        
        for clean_file in clean_files:
            noisy_file = self.noisy_dir / clean_file.name
            if noisy_file.exists():
                pairs.append((clean_file, noisy_file))
            else:
                print(f"Warning: No matching noisy file for {clean_file.name}")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        clean_path, noisy_path = self.file_pairs[idx]
        
        try:
            # Load audio files
            clean_audio, sr = torchaudio.load(clean_path)
            noisy_audio, sr_noisy = torchaudio.load(noisy_path)
            
            # Handle different sample rates
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                clean_audio = resampler(clean_audio)
            
            if sr_noisy != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr_noisy, self.sample_rate)
                noisy_audio = resampler(noisy_audio)
            
            # Convert to mono if necessary
            if clean_audio.shape[0] > 1:
                clean_audio = clean_audio.mean(dim=0, keepdim=True)
            if noisy_audio.shape[0] > 1:
                noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
            
            # Ensure same length
            min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
            clean_audio = clean_audio[:, :min_length]
            noisy_audio = noisy_audio[:, :min_length]
            
            # Apply augmentation (before preprocessing)
            if self.use_augmentation:
                clean_audio, noisy_audio = self._apply_augmentation(clean_audio, noisy_audio)
            
            # Apply preprocessing
            clean_audio, noisy_audio = self._preprocess_audio(clean_audio, noisy_audio)
            
            return {
                'clean': clean_audio,
                'noisy': noisy_audio,
                'filename': clean_path.stem,
                'length': clean_audio.shape[1]
            }
            
        except Exception as e:
            print(f"Error loading {clean_path}: {e}")
            # Return a random other sample
            return self.__getitem__(random.randint(0, len(self.file_pairs) - 1))
    
    def _apply_augmentation(self, clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply audio augmentation"""
        
        # Time stretching
        if self.time_stretch_range[0] != self.time_stretch_range[1]:
            stretch_factor = random.uniform(self.time_stretch_range[0], self.time_stretch_range[1])
            if stretch_factor != 1.0:
                time_stretch = torchaudio.transforms.TimeStretch(
                    n_freq=513, 
                    fixed_rate=stretch_factor
                )
                # Apply STFT, stretch, and inverse STFT
                clean_stft = torch.stft(clean_audio.squeeze(0), n_fft=1024, hop_length=256, return_complex=True)
                noisy_stft = torch.stft(noisy_audio.squeeze(0), n_fft=1024, hop_length=256, return_complex=True)
                
                clean_stretched = time_stretch(clean_stft)
                noisy_stretched = time_stretch(noisy_stft)
                
                clean_audio = torch.istft(clean_stretched, n_fft=1024, hop_length=256).unsqueeze(0)
                noisy_audio = torch.istft(noisy_stretched, n_fft=1024, hop_length=256).unsqueeze(0)
        
        # Pitch shifting
        if self.pitch_shift_range[0] != self.pitch_shift_range[1]:
            pitch_shift_semitones = random.uniform(self.pitch_shift_range[0], self.pitch_shift_range[1])
            if pitch_shift_semitones != 0.0:
                pitch_shift = torchaudio.transforms.PitchShift(
                    sample_rate=self.sample_rate,
                    n_steps=pitch_shift_semitones
                )
                clean_audio = pitch_shift(clean_audio)
                noisy_audio = pitch_shift(noisy_audio)
        
        # Gain adjustment (if enabled)
        if self.gain_range[0] != self.gain_range[1]:
            gain_db = random.uniform(self.gain_range[0], self.gain_range[1])
            if gain_db != 0.0:
                gain_linear = 10 ** (gain_db / 20)
                clean_audio = clean_audio * gain_linear
                noisy_audio = noisy_audio * gain_linear
        
        return clean_audio, noisy_audio
    
    def _preprocess_audio(self, clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess audio (cropping, normalization, etc.)"""
        
        # Cropping for all modes to ensure consistent tensor sizes
        if clean_audio.shape[1] > self.crop_length:
            if self.mode == 'train':
                # Random cropping for training
                start = random.randint(0, clean_audio.shape[1] - self.crop_length)
            else:
                # Center cropping for validation/test
                start = (clean_audio.shape[1] - self.crop_length) // 2
            clean_audio = clean_audio[:, start:start + self.crop_length]
            noisy_audio = noisy_audio[:, start:start + self.crop_length]
        
        # Pad if too short
        if clean_audio.shape[1] < self.crop_length:
            pad_length = self.crop_length - clean_audio.shape[1]
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_length))
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, pad_length))
        
        # Normalize
        clean_audio = self._normalize_audio(clean_audio)
        noisy_audio = self._normalize_audio(noisy_audio)
        
        return clean_audio, noisy_audio
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range"""
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

class CollateFn:
    """Custom collate function for variable-length sequences"""
    
    def __init__(self, max_length: Optional[int] = None):
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        clean_audios = []
        noisy_audios = []
        filenames = []
        lengths = []
        
        for sample in batch:
            clean_audios.append(sample['clean'])
            noisy_audios.append(sample['noisy'])
            filenames.append(sample['filename'])
            lengths.append(sample['length'])
        
        # Stack tensors
        clean_batch = torch.stack(clean_audios, dim=0)
        noisy_batch = torch.stack(noisy_audios, dim=0)
        lengths_batch = torch.tensor(lengths)
        
        return {
            'clean': clean_batch,
            'noisy': noisy_batch,
            'filenames': filenames,
            'lengths': lengths_batch
        }

class AudioDataModule:
    """Data module for handling train/val/test splits"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self):
        """Setup datasets"""
        # Training dataset
        if os.path.exists(self.config['train_data_dir']):
            self.train_dataset = TTSArtifactDataset(
                clean_dir=os.path.join(self.config['train_data_dir'], 'clean'),
                noisy_dir=os.path.join(self.config['train_data_dir'], 'noisy'),
                config=self.config,
                mode='train'
            )
        
        # Validation dataset
        if os.path.exists(self.config['val_data_dir']):
            self.val_dataset = TTSArtifactDataset(
                clean_dir=os.path.join(self.config['val_data_dir'], 'clean'),
                noisy_dir=os.path.join(self.config['val_data_dir'], 'noisy'),
                config=self.config,
                mode='val'
            )
        
        # Test dataset
        test_dir = self.config.get('test_data_dir')
        if test_dir and os.path.exists(test_dir):
            self.test_dataset = TTSArtifactDataset(
                clean_dir=os.path.join(test_dir, 'clean'),
                noisy_dir=os.path.join(test_dir, 'noisy'),
                config=self.config,
                mode='test'
            )
    
    def train_dataloader(self):
        """Create training dataloader"""
        if self.train_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            collate_fn=CollateFn()
        )
    
    def val_dataloader(self):
        """Create validation dataloader"""
        if self.val_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True),
            collate_fn=CollateFn()
        )
    
    def test_dataloader(self):
        """Create test dataloader"""
        if self.test_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
            collate_fn=CollateFn()
        ) 