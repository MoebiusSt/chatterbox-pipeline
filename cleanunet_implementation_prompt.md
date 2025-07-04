# Agent-Anweisung: CleanUNet TTS-Artefakt-Entfernung implementieren

## Projektziel
Erstelle ein vollständiges CleanUNet-basiertes System namens "Chatterbox-CleanUNet" zur selektiven Entfernung von TTS-Artefakten aus Audiodateien. Das System soll trainierbar und einsatzbereit sein für die Behandlung wiederkehrender Säusel- und Sirrgeräusche in Voice-Cloning-generierten Audiodateien.
Das System ist von "TTS-Pipeline-enhanced" unabhängig, soll aber hier residieren.

## Verzeichnisstruktur erstellen

Erstelle im TTS-Pipeline-enhanced Projekt folgende Struktur:

```
TTS-Pipeline-enhanced/
└── Chatterbox-CleanUNet/
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    ├── config/
    │   ├── train_config.yaml
    │   ├── model_config.yaml
    │   └── inference_config.yaml
    ├── src/
    │   ├── __init__.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── cleanunet.py
    │   │   ├── attention.py
    │   │   └── loss.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── dataset.py
    │   │   ├── preprocessor.py
    │   │   └── augmentation.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── trainer.py
    │   │   ├── validator.py
    │   │   └── checkpoints.py
    │   ├── inference/
    │   │   ├── __init__.py
    │   │   ├── enhancer.py
    │   │   └── batch_processor.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── audio_utils.py
    │       ├── metrics.py
    │       └── visualization.py
    ├── scripts/
    │   ├── train.py
    │   ├── enhance_audio.py
    │   ├── evaluate_model.py
    │   ├── prepare_dataset.py
    │   └── download_dependencies.py
    ├── data/
    │   ├── raw/
    │   │   ├── clean/
    │   │   └── noisy/
    │   ├── processed/
    │   │   ├── train/
    │   │   ├── validation/
    │   │   └── test/
    │   └── examples/
    ├── models/
    │   ├── checkpoints/
    │   ├── pretrained/
    │   └── final/
    ├── outputs/
    │   ├── enhanced_audio/
    │   ├── training_logs/
    │   └── evaluation_results/
    └── tests/
        ├── __init__.py
        ├── test_model.py
        ├── test_data.py
        └── test_inference.py
```

## 1. Basis-Setup und Dependencies

### requirements.txt erstellen:
```txt
torch>=1.12.0
torchaudio>=0.12.0
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.9.0
pesq>=0.0.3
pystoi>=0.3.3
PyYAML>=6.0
tqdm>=4.64.0
matplotlib>=3.5.0
tensorboard>=2.8.0
soundfile>=0.10.0
```

### setup.py erstellen:
```python
from setuptools import setup, find_packages

setup(
    name="chatterbox-cleanunet",
    version="1.0.0",
    description="CleanUNet-based TTS artifact removal system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "pesq>=0.0.3",
        "pystoi>=0.3.3",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "soundfile>=0.10.0",
    ],
    entry_points={
        "console_scripts": [
            "cleanunet-train=scripts.train:main",
            "cleanunet-enhance=scripts.enhance_audio:main",
            "cleanunet-evaluate=scripts.evaluate_model:main",
        ],
    },
)
```

## 2. Konfigurationsdateien

### config/model_config.yaml:
```yaml
model:
  # Audio parameters
  sample_rate: 16000
  n_fft: 512
  hop_length: 128
  win_length: 512
  
  # Model architecture
  encoder_layers: 8
  decoder_layers: 8
  encoder_channels: [64, 128, 256, 512, 512, 512, 512, 512]
  decoder_channels: [512, 512, 512, 512, 256, 128, 64, 32]
  
  # Self-attention parameters
  attention_heads: 8
  attention_dim: 512
  attention_layers: 4
  attention_dropout: 0.1
  
  # Other parameters
  kernel_size: 3
  stride: 2
  activation: "LeakyReLU"
  normalization: "BatchNorm1d"

loss:
  l1_weight: 1.0
  stft_weight: 0.5
  high_band_stft_weight: 0.3
  high_band_freq_min: 4000
  high_band_freq_max: 8000
```

### config/train_config.yaml:
```yaml
training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.00001
  gradient_clip_norm: 5.0
  
  # Scheduler
  scheduler: "ReduceLROnPlateau"
  scheduler_patience: 10
  scheduler_factor: 0.5
  
  # Validation
  validation_freq: 1
  save_freq: 5
  early_stopping_patience: 20
  
  # Data
  train_data_dir: "data/processed/train"
  val_data_dir: "data/processed/validation"
  num_workers: 4
  pin_memory: true
  
  # Augmentation
  use_augmentation: true
  time_stretch_range: [0.9, 1.1]
  pitch_shift_range: [-2, 2]
  noise_snr_range: [10, 30]

hardware:
  device: "cuda"
  mixed_precision: true
  compile_model: false
```

### config/inference_config.yaml:
```yaml
inference:
  model_path: "models/final/cleanunet_best.pth"
  device: "cuda"
  batch_size: 1
  
  # Audio processing
  sample_rate: 16000
  chunk_size: 32768  # 2 seconds at 16kHz
  overlap: 0.25      # 25% overlap between chunks
  
  # Output
  output_format: "wav"
  output_sample_rate: 16000
  normalize_output: true
```

## 3. Core Model Implementation

### src/models/cleanunet.py:
Implementiere die komplette CleanUNet Architektur basierend auf der NVIDIA-Implementierung:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttentionBlock

class CleanUNet(nn.Module):
    """
    CleanUNet model for TTS artifact removal
    Based on: Speech Denoising in the Waveform Domain with Self-Attention
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in config['encoder_channels']:
            self.encoder_layers.append(
                EncoderBlock(in_channels, out_channels, config)
            )
            in_channels = out_channels
        
        # Self-attention blocks
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(
                config['attention_dim'],
                config['attention_heads'],
                config['attention_dropout']
            )
            for _ in range(config['attention_layers'])
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        channels = config['decoder_channels']
        for i, out_channels in enumerate(channels):
            skip_channels = config['encoder_channels'][-(i+1)]
            in_channels = channels[i-1] if i > 0 else config['encoder_channels'][-1]
            self.decoder_layers.append(
                DecoderBlock(in_channels + skip_channels, out_channels, config)
            )
        
        # Final output layer
        self.output_conv = nn.Conv1d(channels[-1], 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        for encoder in self.encoder_layers:
            x = encoder(x)
            skip_connections.append(x)
        
        # Self-attention in bottleneck
        for attention in self.attention_blocks:
            x = attention(x)
        
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            x = decoder(torch.cat([x, skip], dim=1))
        
        # Final output
        x = self.output_conv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['kernel_size']//2
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=config['kernel_size']//2,
            output_padding=config['stride']-1
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.activation(self.norm(self.conv_transpose(x)))
```

### src/models/attention.py:
```python
import torch
import torch.nn as nn
import math

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        # Convert to (batch, time, channels) for attention
        x = x.transpose(1, 2)
        
        # Self-attention with residual connection
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Convert back to (batch, channels, time)
        return x.transpose(1, 2)
```

## 4. Datenverwaltung

### src/data/dataset.py:
```python
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import random
from pathlib import Path

class TTSArtifactDataset(Dataset):
    """Dataset for TTS artifact removal training"""
    
    def __init__(self, clean_dir, noisy_dir, config, mode='train'):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.config = config
        self.mode = mode
        
        # Get matching clean/noisy pairs
        self.file_pairs = self._get_file_pairs()
        
    def _get_file_pairs(self):
        clean_files = list(self.clean_dir.glob("*.wav"))
        pairs = []
        
        for clean_file in clean_files:
            noisy_file = self.noisy_dir / clean_file.name
            if noisy_file.exists():
                pairs.append((clean_file, noisy_file))
        
        return pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        clean_path, noisy_path = self.file_pairs[idx]
        
        # Load audio
        clean_audio, sr = torchaudio.load(clean_path)
        noisy_audio, sr = torchaudio.load(noisy_path)
        
        # Resample if necessary
        if sr != self.config['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, self.config['sample_rate'])
            clean_audio = resampler(clean_audio)
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
        
        # Random cropping for training
        if self.mode == 'train' and min_length > self.config.get('crop_length', 32768):
            crop_length = self.config.get('crop_length', 32768)
            start = random.randint(0, min_length - crop_length)
            clean_audio = clean_audio[:, start:start + crop_length]
            noisy_audio = noisy_audio[:, start:start + crop_length]
        
        return {
            'clean': clean_audio,
            'noisy': noisy_audio,
            'filename': clean_path.stem
        }
```

## 5. Training Pipeline

### scripts/train.py:
```python
#!/usr/bin/env python3
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cleanunet import CleanUNet
from models.loss import CleanUNetLoss
from data.dataset import TTSArtifactDataset
from training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description='Train CleanUNet for TTS artifact removal')
    parser.add_argument('--config', default='config/train_config.yaml', help='Training config file')
    parser.add_argument('--model_config', default='config/model_config.yaml', help='Model config file')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(train_config['hardware']['device'])
    
    # Create datasets
    train_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(train_config['training']['train_data_dir'], 'clean'),
        noisy_dir=os.path.join(train_config['training']['train_data_dir'], 'noisy'),
        config=model_config['model'],
        mode='train'
    )
    
    val_dataset = TTSArtifactDataset(
        clean_dir=os.path.join(train_config['training']['val_data_dir'], 'clean'),
        noisy_dir=os.path.join(train_config['training']['val_data_dir'], 'noisy'),
        config=model_config['model'],
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=True,
        num_workers=train_config['training']['num_workers'],
        pin_memory=train_config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['training']['batch_size'],
        shuffle=False,
        num_workers=train_config['training']['num_workers'],
        pin_memory=train_config['training']['pin_memory']
    )
    
    # Create model
    model = CleanUNet(model_config['model']).to(device)
    
    # Create loss function
    criterion = CleanUNetLoss(model_config['loss']).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config['training']['learning_rate'],
        weight_decay=train_config['training']['weight_decay']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()
```

## 6. Inference Tool

### scripts/enhance_audio.py:
```python
#!/usr/bin/env python3
import argparse
import yaml
import torch
import torchaudio
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cleanunet import CleanUNet
from inference.enhancer import AudioEnhancer

def main():
    parser = argparse.ArgumentParser(description='Enhance audio using trained CleanUNet')
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('output', help='Output audio file or directory')
    parser.add_argument('--config', default='config/inference_config.yaml', help='Inference config')
    parser.add_argument('--model_config', default='config/model_config.yaml', help='Model config')
    parser.add_argument('--model', help='Model checkpoint path (overrides config)')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    args = parser.parse_args()
    
    # Load configurations
    with open(args.config, 'r') as f:
        inference_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Override model path if specified
    if args.model:
        inference_config['inference']['model_path'] = args.model
    
    # Create enhancer
    enhancer = AudioEnhancer(inference_config, model_config)
    
    # Process audio
    if args.batch:
        enhancer.enhance_directory(args.input, args.output)
    else:
        enhancer.enhance_file(args.input, args.output)
    
    print(f"Enhancement complete! Output saved to: {args.output}")

if __name__ == '__main__':
    main()
```

## 7. Zusätzliche Tools

### scripts/prepare_dataset.py:
Erstelle ein Tool zum Vorbereiten von Trainingsdaten aus Clean/Noisy Audio-Paaren:

```python
#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
import torchaudio
import random

def prepare_dataset(clean_dir, noisy_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Prepare dataset by splitting into train/validation/test sets
    """
    # Implementation details...
    pass

if __name__ == '__main__':
    # Argument parsing and main logic...
    pass
```

### scripts/evaluate_model.py:
Erstelle ein Evaluations-Tool:

```python
#!/usr/bin/env python3
import argparse
import yaml
import torch
from pesq import pesq
from pystoi import stoi
import numpy as np

def evaluate_model(model_path, test_data_path, config):
    """
    Evaluate trained model on test dataset
    """
    # Implementation details...
    pass

if __name__ == '__main__':
    # Argument parsing and main logic...
    pass
```

## 8. CLI Interface und Installation

Erstelle ein CLI-Interface das folgende Befehle unterstützt:

```bash
# Installation
cd TTS-Pipeline-enhanced/Chatterbox-CleanUNet
pip install -e .

# Training
cleanunet-train --config config/train_config.yaml

# Enhancement
cleanunet-enhance input.wav output.wav
cleanunet-enhance --batch input_dir/ output_dir/

# Evaluation
cleanunet-evaluate --model models/final/cleanunet_best.pth --test-data data/processed/test/
```

## 9. README.md erstellen

Erstelle eine umfassende README.md mit:
- Installation instructions
- Quick start guide
- Training procedure
- Usage examples
- Configuration options
- Troubleshooting

## 10. Tests implementieren

Erstelle Unit-Tests für alle wichtigen Komponenten:
- Model architecture tests
- Data loading tests
- Training pipeline tests
- Inference tests

## Erfolgs-Kriterien

Das fertige System soll:
1. ✅ Eigenständig lauffähig sein
2. ✅ Training auf eigenen TTS-Artefakt-Daten ermöglichen
3. ✅ Audio-Dateien batch-weise verarbeiten können
4. ✅ Evaluation-Metriken (PESQ, STOI) bereitstellen
5. ✅ RTX 4080-optimiert sein (16GB VRAM)
6. ✅ Vollständige CLI-Integration bieten
7. ✅ Reproduzierbare Ergebnisse liefern

## Implementierungs-Reihenfolge

1. **Setup**: Verzeichnisstruktur und Dependencies
2. **Core Model**: CleanUNet Architektur implementieren
3. **Data Pipeline**: Dataset und DataLoader erstellen
4. **Training**: Training-Loop und Checkpointing
5. **Inference**: Audio-Enhancement-Tool
6. **CLI**: Command-line Interface
7. **Tests**: Unit-Tests und Integration-Tests
8. **Documentation**: README und Code-Dokumentation

Beginne mit Schritt 1 und arbeite systematisch durch die Liste. Jeder Schritt sollte vollständig getestet werden bevor zum nächsten übergegangen wird.