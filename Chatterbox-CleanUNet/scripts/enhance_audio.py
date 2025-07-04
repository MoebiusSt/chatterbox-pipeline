#!/usr/bin/env python3
"""
Audio enhancement script for CleanUNet TTS artifact removal
"""
import argparse
import yaml
import torch
import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.enhancer import AudioEnhancer
from utils.audio_utils import load_audio

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhance audio using trained CleanUNet')
    
    # Input/Output
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('output', help='Output audio file or directory')
    
    # Configuration
    parser.add_argument('--config', default='config/inference_config.yaml', 
                       help='Inference configuration file')
    parser.add_argument('--model_config', default='config/model_config.yaml', 
                       help='Model configuration file')
    parser.add_argument('--model', help='Model checkpoint path (overrides config)')
    
    # Processing options
    parser.add_argument('--batch', action='store_true', 
                       help='Process directory in batch mode')
    parser.add_argument('--pattern', default='*.wav', 
                       help='File pattern for batch processing (default: *.wav)')
    parser.add_argument('--recursive', action='store_true', 
                       help='Process directories recursively')
    
    # Audio options
    parser.add_argument('--sample_rate', type=int, 
                       help='Override sample rate')
    parser.add_argument('--chunk_size', type=int, 
                       help='Override chunk size for processing')
    parser.add_argument('--no_normalize', action='store_true', 
                       help='Disable output normalization')
    
    # Hardware options
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU processing')
    
    # Output options
    parser.add_argument('--suffix', default='_enhanced', 
                       help='Suffix for enhanced files (default: _enhanced)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing files')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    # Quality evaluation
    parser.add_argument('--reference', help='Reference clean audio for quality evaluation')
    parser.add_argument('--metrics', action='store_true', 
                       help='Compute quality metrics (requires reference)')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def override_config(config: dict, args) -> dict:
    """Override configuration with command line arguments"""
    if args.model:
        config['inference']['model_path'] = args.model
    
    if args.device:
        config['inference']['device'] = args.device
    elif args.cpu:
        config['inference']['device'] = 'cpu'
    
    if args.sample_rate:
        config['inference']['sample_rate'] = args.sample_rate
    
    if args.chunk_size:
        config['inference']['chunk_size'] = args.chunk_size
    
    if args.no_normalize:
        config['inference']['normalize_output'] = False
    
    return config

def validate_inputs(args):
    """Validate input arguments"""
    input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if args.batch and not input_path.is_dir():
        raise ValueError("Batch mode requires input to be a directory")
    
    if not args.batch and input_path.is_dir():
        raise ValueError("Single file mode requires input to be a file. Use --batch for directories")
    
    # Check model file exists
    config_path = args.config
    if os.path.exists(config_path):
        config = load_config(config_path)
        model_path = args.model or config['inference']['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

def process_single_file(enhancer: AudioEnhancer, input_path: str, output_path: str, 
                       args) -> dict:
    """Process a single audio file"""
    if args.verbose:
        print(f"Processing: {input_path} -> {output_path}")
    
    # Check if output exists and handle overwrite
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output file exists, skipping: {output_path}")
        print("Use --overwrite to overwrite existing files")
        return {}
    
    # Process with or without reference
    if args.reference:
        stats = enhancer.enhance_with_comparison(
            input_path, 
            str(Path(output_path).parent), 
            args.reference
        )
    else:
        stats = enhancer.enhance_file(input_path, output_path)
    
    return stats

def process_batch(enhancer: AudioEnhancer, input_dir: str, output_dir: str, 
                 args) -> dict:
    """Process multiple audio files in batch"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    if args.recursive:
        audio_files = list(input_path.rglob(args.pattern))
    else:
        audio_files = list(input_path.glob(args.pattern))
    
    if not audio_files:
        print(f"No audio files found in {input_dir} matching pattern {args.pattern}")
        return {}
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    total_stats = {
        'total_files': len(audio_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_processing_time': 0,
        'total_audio_length': 0,
        'failed_file_list': []
    }
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            # Determine output path
            if args.recursive:
                # Preserve directory structure
                rel_path = audio_file.relative_to(input_path)
                output_file = output_path / rel_path.parent / f"{rel_path.stem}{args.suffix}{rel_path.suffix}"
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = output_path / f"{audio_file.stem}{args.suffix}{audio_file.suffix}"
            
            # Check if output exists
            if output_file.exists() and not args.overwrite:
                if args.verbose:
                    print(f"Skipping existing file: {output_file}")
                continue
            
            # Process file
            if args.verbose:
                print(f"[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            stats = enhancer.enhance_file(str(audio_file), str(output_file))
            
            # Update totals
            total_stats['processed_files'] += 1
            total_stats['total_processing_time'] += stats['processing_time']
            total_stats['total_audio_length'] += stats['audio_length']
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            total_stats['failed_files'] += 1
            total_stats['failed_file_list'].append(str(audio_file))
    
    # Calculate overall statistics
    if total_stats['total_audio_length'] > 0:
        total_stats['average_real_time_factor'] = (
            total_stats['total_processing_time'] / total_stats['total_audio_length']
        )
    else:
        total_stats['average_real_time_factor'] = 0
    
    return total_stats

def print_summary(stats: dict, batch_mode: bool):
    """Print processing summary"""
    if not stats:
        return
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    
    if batch_mode:
        print(f"Total files: {stats['total_files']}")
        print(f"Processed: {stats['processed_files']}")
        print(f"Failed: {stats['failed_files']}")
        print(f"Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"Total audio length: {stats['total_audio_length']:.2f}s")
        print(f"Average real-time factor: {stats['average_real_time_factor']:.2f}x")
        
        if stats['failed_file_list']:
            print(f"\nFailed files:")
            for failed_file in stats['failed_file_list']:
                print(f"  - {failed_file}")
    else:
        print(f"Processing time: {stats['processing_time']:.2f}s")
        print(f"Audio length: {stats['audio_length']:.2f}s")
        print(f"Real-time factor: {stats['real_time_factor']:.2f}x")
        
        # Print quality metrics if available
        if 'pesq' in stats:
            print(f"\nQuality Metrics:")
            print(f"PESQ: {stats['pesq']:.3f}")
            print(f"STOI: {stats['stoi']:.3f}")
            print(f"ESTOI: {stats['estoi']:.3f}")
            print(f"SI-SNR: {stats['si_snr']:.3f} dB")

def main():
    """Main enhancement function"""
    args = parse_args()
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Load configurations
        if args.verbose:
            print("Loading configurations...")
        
        inference_config = load_config(args.config)
        model_config = load_config(args.model_config)
        
        # Override configuration with command line arguments
        inference_config = override_config(inference_config, args)
        
        # Create enhancer
        if args.verbose:
            print("Initializing audio enhancer...")
        
        enhancer = AudioEnhancer(inference_config, model_config)
        
        # Print model info
        if args.verbose:
            model_info = enhancer.get_model_info()
            print(f"Model: {model_info['model_path']}")
            print(f"Device: {model_info['device']}")
            print(f"Parameters: {model_info['total_parameters']:,}")
            print(f"Model size: {model_info['model_size_mb']:.1f} MB")
        
        # Process audio
        start_time = time.time()
        
        if args.batch:
            stats = process_batch(enhancer, args.input, args.output, args)
        else:
            stats = process_single_file(enhancer, args.input, args.output, args)
        
        total_time = time.time() - start_time
        
        # Print summary
        if args.verbose:
            print_summary(stats, args.batch)
            print(f"Total execution time: {total_time:.2f}s")
        
        print("\nEnhancement completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 