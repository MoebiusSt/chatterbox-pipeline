#!/usr/bin/env python3
"""
Model evaluation script for CleanUNet TTS artifact removal
Evaluates trained model on test dataset and generates quality reports
"""
import argparse
import yaml
import torch
import os
import sys
from pathlib import Path
import json
import time
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.enhancer import AudioEnhancer
from utils.metrics import MetricsCalculator, compute_all_metrics
from utils.audio_utils import load_audio, save_audio
from data.dataset import TTSArtifactDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained CleanUNet model')
    
    # Model and data
    parser.add_argument('--model', required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', required=True, 
                       help='Path to test dataset directory')
    parser.add_argument('--output_dir', default='outputs/evaluation', 
                       help='Output directory for evaluation results')
    
    # Configuration
    parser.add_argument('--model_config', default='config/model_config.yaml', 
                       help='Model configuration file')
    parser.add_argument('--inference_config', default='config/inference_config.yaml', 
                       help='Inference configuration file')
    
    # Evaluation options
    parser.add_argument('--save_examples', type=int, default=10, 
                       help='Number of example outputs to save (default: 10)')
    parser.add_argument('--compute_detailed_metrics', action='store_true', 
                       help='Compute additional detailed metrics')
    parser.add_argument('--create_plots', action='store_true', 
                       help='Create visualization plots')
    
    # Hardware options
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size for evaluation')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--save_enhanced_audio', action='store_true', 
                       help='Save all enhanced audio files')
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_output_directory(output_dir: Path):
    """Create output directory structure"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = ['enhanced_audio', 'examples', 'plots', 'reports']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(exist_ok=True)

def evaluate_dataset(enhancer: AudioEnhancer, test_dataset, output_dir: Path, 
                     args) -> dict:
    """Evaluate model on entire test dataset"""
    
    metrics_calculator = MetricsCalculator(sample_rate=24000)
    results = {
        'total_files': len(test_dataset),
        'processed_files': 0,
        'failed_files': 0,
        'total_processing_time': 0,
        'total_audio_length': 0,
        'individual_results': []
    }
    
    print(f"Evaluating model on {len(test_dataset)} test samples...")
    
    # Process each sample
    for i, sample in enumerate(tqdm(test_dataset, desc="Evaluating")):
        try:
            clean_audio = sample['clean']
            noisy_audio = sample['noisy']
            filename = sample['filename']
            
            start_time = time.time()
            
            # Enhance audio
            enhanced_audio = enhancer.enhance_audio(noisy_audio)
            
            processing_time = time.time() - start_time
            audio_length = clean_audio.shape[1] / 24000  # Assuming 24kHz
            
            # Compute metrics
            metrics = compute_all_metrics(clean_audio, enhanced_audio, 24000)
            
            # Update overall metrics
            metrics_calculator.update(clean_audio, enhanced_audio)
            
            # Store individual result
            individual_result = {
                'filename': filename,
                'processing_time': processing_time,
                'audio_length': audio_length,
                'real_time_factor': processing_time / audio_length if audio_length > 0 else 0,
                **metrics
            }
            results['individual_results'].append(individual_result)
            
            # Update totals
            results['processed_files'] += 1
            results['total_processing_time'] += processing_time
            results['total_audio_length'] += audio_length
            
            # Save examples
            if args.save_examples > 0 and i < args.save_examples:
                example_dir = output_dir / 'examples'
                save_audio(clean_audio, str(example_dir / f"{filename}_clean.wav"), 24000)
                save_audio(noisy_audio, str(example_dir / f"{filename}_noisy.wav"), 24000)
                save_audio(enhanced_audio, str(example_dir / f"{filename}_enhanced.wav"), 24000)
            
            # Save all enhanced audio if requested
            if args.save_enhanced_audio:
                enhanced_dir = output_dir / 'enhanced_audio'
                save_audio(enhanced_audio, str(enhanced_dir / f"{filename}_enhanced.wav"), 24000)
            
            if args.verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} files...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results['failed_files'] += 1
    
    # Compute overall metrics
    overall_metrics = metrics_calculator.compute_average()
    results['overall_metrics'] = overall_metrics
    
    # Calculate overall statistics
    if results['total_audio_length'] > 0:
        results['average_real_time_factor'] = (
            results['total_processing_time'] / results['total_audio_length']
        )
    else:
        results['average_real_time_factor'] = 0
    
    return results

def create_evaluation_report(results: dict, output_dir: Path, model_path: str):
    """Create detailed evaluation report"""
    
    report_path = output_dir / 'reports' / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("CleanUNet Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Model information
        f.write(f"Model: {model_path}\n")
        f.write(f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset statistics
        f.write("Dataset Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total files: {results['total_files']}\n")
        f.write(f"Successfully processed: {results['processed_files']}\n")
        f.write(f"Failed: {results['failed_files']}\n")
        f.write(f"Success rate: {results['processed_files']/results['total_files']:.1%}\n\n")
        
        # Performance statistics
        f.write("Performance Statistics:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total processing time: {results['total_processing_time']:.2f}s\n")
        f.write(f"Total audio length: {results['total_audio_length']:.2f}s\n")
        f.write(f"Average real-time factor: {results['average_real_time_factor']:.3f}x\n\n")
        
        # Quality metrics
        f.write("Quality Metrics (Average ± Std):\n")
        f.write("-" * 35 + "\n")
        metrics = results['overall_metrics']
        f.write(f"PESQ:      {metrics['pesq']:.3f} ± {metrics['pesq_std']:.3f}\n")
        f.write(f"STOI:      {metrics['stoi']:.3f} ± {metrics['stoi_std']:.3f}\n")
        f.write(f"ESTOI:     {metrics['estoi']:.3f} ± {metrics['estoi_std']:.3f}\n")
        f.write(f"SNR:       {metrics['snr']:.3f} ± {metrics['snr_std']:.3f} dB\n")
        f.write(f"SI-SNR:    {metrics['si_snr']:.3f} ± {metrics['si_snr_std']:.3f} dB\n")
        f.write(f"LSD:       {metrics['lsd']:.3f} ± {metrics['lsd_std']:.3f}\n")
        f.write(f"Spec Conv: {metrics['spectral_convergence']:.3f} ± {metrics['spectral_convergence_std']:.3f}\n")
        f.write(f"MCD:       {metrics['mcd']:.3f} ± {metrics['mcd_std']:.3f}\n\n")
        
        # Best and worst performing files
        if results['individual_results']:
            individual = results['individual_results']
            
            # Sort by PESQ score
            sorted_by_pesq = sorted(individual, key=lambda x: x.get('pesq', 0), reverse=True)
            
            f.write("Best Performing Files (by PESQ):\n")
            f.write("-" * 35 + "\n")
            for i, result in enumerate(sorted_by_pesq[:5]):
                f.write(f"{i+1}. {result['filename']}: PESQ={result.get('pesq', 'N/A'):.3f}, STOI={result.get('stoi', 'N/A'):.3f}\n")
            
            f.write("\nWorst Performing Files (by PESQ):\n")
            f.write("-" * 36 + "\n")
            for i, result in enumerate(sorted_by_pesq[-5:]):
                f.write(f"{i+1}. {result['filename']}: PESQ={result.get('pesq', 'N/A'):.3f}, STOI={result.get('stoi', 'N/A'):.3f}\n")

def save_detailed_results(results: dict, output_dir: Path):
    """Save detailed results to JSON"""
    json_path = output_dir / 'reports' / 'detailed_results.json'
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

def create_summary_plots(results: dict, output_dir: Path):
    """Create summary visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        plots_dir = output_dir / 'plots'
        
        # Get individual results
        individual = results['individual_results']
        if not individual:
            return
        
        # Extract metrics
        pesq_scores = [r.get('pesq', 0) for r in individual]
        stoi_scores = [r.get('stoi', 0) for r in individual]
        snr_scores = [r.get('snr', 0) for r in individual]
        rtf_scores = [r.get('real_time_factor', 0) for r in individual]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('CleanUNet Evaluation Results', fontsize=16)
        
        # PESQ histogram
        axes[0, 0].hist(pesq_scores, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('PESQ Score Distribution')
        axes[0, 0].set_xlabel('PESQ Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(pesq_scores), color='red', linestyle='--', label=f'Mean: {np.mean(pesq_scores):.3f}')
        axes[0, 0].legend()
        
        # STOI histogram
        axes[0, 1].hist(stoi_scores, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('STOI Score Distribution')
        axes[0, 1].set_xlabel('STOI Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(stoi_scores), color='red', linestyle='--', label=f'Mean: {np.mean(stoi_scores):.3f}')
        axes[0, 1].legend()
        
        # SNR histogram
        axes[1, 0].hist(snr_scores, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('SNR Distribution')
        axes[1, 0].set_xlabel('SNR (dB)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(snr_scores), color='red', linestyle='--', label=f'Mean: {np.mean(snr_scores):.1f} dB')
        axes[1, 0].legend()
        
        # Real-time factor histogram
        axes[1, 1].hist(rtf_scores, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_title('Real-time Factor Distribution')
        axes[1, 1].set_xlabel('Real-time Factor')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(rtf_scores), color='red', linestyle='--', label=f'Mean: {np.mean(rtf_scores):.3f}x')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pesq_scores, stoi_scores, alpha=0.6)
        ax.set_xlabel('PESQ Score')
        ax.set_ylabel('STOI Score')
        ax.set_title('PESQ vs STOI Correlation')
        
        # Add correlation coefficient
        correlation = np.corrcoef(pesq_scores, stoi_scores)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(plots_dir / 'pesq_stoi_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plots_dir}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot creation")
    except Exception as e:
        print(f"Error creating plots: {e}")

def main():
    """Main evaluation function"""
    args = parse_args()
    
    try:
        # Load configurations
        if args.verbose:
            print("Loading configurations...")
        
        model_config = load_config(args.model_config)
        inference_config = load_config(args.inference_config)
        
        # Override model path
        inference_config['inference']['model_path'] = args.model
        
        # Override device if specified
        if args.device:
            inference_config['inference']['device'] = args.device
        
        # Setup output directory
        output_dir = Path(args.output_dir)
        setup_output_directory(output_dir)
        
        # Create enhancer
        if args.verbose:
            print("Loading model...")
        
        enhancer = AudioEnhancer(inference_config, model_config)
        
        # Print model info
        if args.verbose:
            model_info = enhancer.get_model_info()
            print(f"Model: {model_info['model_path']}")
            print(f"Device: {model_info['device']}")
            print(f"Parameters: {model_info['total_parameters']:,}")
        
        # Load test dataset
        if args.verbose:
            print("Loading test dataset...")
        
        test_dataset = TTSArtifactDataset(
            clean_dir=os.path.join(args.test_data, 'clean'),
            noisy_dir=os.path.join(args.test_data, 'noisy'),
            config=model_config['model'],
            mode='test'
        )
        
        # Run evaluation
        start_time = time.time()
        results = evaluate_dataset(enhancer, test_dataset, output_dir, args)
        total_evaluation_time = time.time() - start_time
        
        # Add evaluation metadata
        results['evaluation_metadata'] = {
            'model_path': args.model,
            'test_data_path': args.test_data,
            'evaluation_time': total_evaluation_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create reports
        if args.verbose:
            print("Generating reports...")
        
        create_evaluation_report(results, output_dir, args.model)
        save_detailed_results(results, output_dir)
        
        # Create plots if requested
        if args.create_plots:
            create_summary_plots(results, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Test files: {results['processed_files']}/{results['total_files']}")
        print(f"Evaluation time: {total_evaluation_time:.1f}s")
        print(f"Average real-time factor: {results['average_real_time_factor']:.3f}x")
        
        metrics = results['overall_metrics']
        print(f"\nQuality Metrics:")
        print(f"  PESQ:  {metrics['pesq']:.3f} ± {metrics['pesq_std']:.3f}")
        print(f"  STOI:  {metrics['stoi']:.3f} ± {metrics['stoi_std']:.3f}")
        print(f"  SI-SNR: {metrics['si_snr']:.1f} ± {metrics['si_snr_std']:.1f} dB")
        
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"📄 Report: {output_dir}/reports/evaluation_report.txt")
        print(f"📊 Data: {output_dir}/reports/detailed_results.json")
        
        if args.create_plots:
            print(f"📈 Plots: {output_dir}/plots/")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 