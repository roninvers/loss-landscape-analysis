import torch
import yaml
import os
import json
import numpy as np
from datetime import datetime

from data_loader import DataLoaderManager
from model_builder import get_model
from training import Trainer
from landscape_analysis import LandscapeAnalyzer
from metrics_computer import MetricsComputer
from visualization import LandscapeVisualizer


class LossLandscapePipeline:
    """Complete pipeline for loss landscape analysis."""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize pipeline from configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        if self.config['device'] == 'mps':
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        elif self.config['device'] == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        
        self.dtype = getattr(torch, self.config['dtype'])
        
        # Set random seeds
        torch.manual_seed(self.config['random_seed'])
        np.random.seed(self.config['numpy_seed'])
        
        # Create result directories
        os.makedirs(self.config['output']['results_dir'], exist_ok=True)
        os.makedirs(self.config['output']['models_dir'], exist_ok=True)
        
        print(f"✓ Device: {self.device}")
        print(f"✓ Data type: {self.dtype}\n")


    def run_all_experiments(self, model_name='resnet20', use_skip=False, 
                       use_batchnorm=True, dataset='cifar10'):
        """Run single experiment with corrected in_channels detection."""
        
        experiment_name = f"{model_name}_skip{use_skip}_bn{use_batchnorm}"
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*70}\n")
        
        # 1. Load data
        print("[1/6] Loading dataset...")
        data_manager = DataLoaderManager(
            dataset_name=dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'],
            augmentation=self.config['dataset']['augmentation']
        )
        train_loader, test_loader, train_size, test_size = data_manager.get_loaders()
        print(f"      Training samples: {train_size}")
        print(f"      Test samples: {test_size}\n")
        
        # Get a sample batch to determine input channels
        sample_images, sample_labels = next(iter(train_loader))
        num_input_channels = sample_images.shape[1]  # Get channels from actual data
        print(f"      Input channels detected: {num_input_channels}\n")
        
        # 2. Create model with CORRECT input channels
        print("[2/6] Creating model...")
        num_classes = 10
        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            use_skip=use_skip,
            use_batchnorm=use_batchnorm,
            device=self.device,
            in_channels=num_input_channels  # CORRECTLY DETECTED FROM DATA!
        )
        print()
        
        # 3. Train model
        print("[3/6] Training model...")
        trainer = Trainer(model, device=self.device, dtype=self.dtype)
        training_results = trainer.train(
            train_loader, test_loader,
            epochs=self.config['training']['epochs'],
            learning_rate=self.config['training']['learning_rate'],
            optimizer_name=self.config['training']['optimizer']
        )
        print(f"      Final test accuracy: {training_results['final_test_acc']:.2f}%")
        print(f"      Generalization gap: {training_results['generalization_gap']:.2f}%\n")
        
        # Save model
        model_path = os.path.join(
            self.config['output']['models_dir'],
            f"{experiment_name}.pt"
        )
        torch.save(model.state_dict(), model_path)
        print(f"      Model saved to {model_path}\n")
        
        # 4. Analyze landscape
        print("[4/6] Analyzing loss landscape...")
        analyzer = LandscapeAnalyzer(model, train_loader, test_loader,
                                    device=self.device, dtype=self.dtype)
        landscape_data = analyzer.compute_2d_landscape(
            num_samples=self.config['landscape']['num_samples']
        )
        landscape_stats_train = analyzer.get_landscape_statistics(landscape_data, 'train')
        landscape_stats_test = analyzer.get_landscape_statistics(landscape_data, 'test')
        print(f"      Train loss range: [{landscape_stats_train['min']:.4f}, {landscape_stats_train['max']:.4f}]")
        print(f"      Test loss range: [{landscape_stats_test['min']:.4f}, {landscape_stats_test['max']:.4f}]\n")
        
        # 5. Compute metrics
        print("[5/6] Computing geometric metrics...")
        metrics_computer = MetricsComputer(model, train_loader,
                                          device=self.device, dtype=self.dtype)
        sharpness_metrics = metrics_computer.compute_sharpness_metrics()
        print(f"      Top eigenvalue: {sharpness_metrics['top_eigenvalue']:.6f}")
        print(f"      Condition number: {sharpness_metrics['condition_number']:.2f}")
        print(f"      Gradient norm: {sharpness_metrics['gradient_norm']:.6f}\n")
        
        # 6. Visualize
        print("[6/6] Generating visualizations...")
        visualizer = LandscapeVisualizer(save_dir=self.config['output']['results_dir'])
        
        title = f"{model_name} (skip={use_skip}, bn={use_batchnorm})"
        
        visualizer.plot_2d_landscape(
            landscape_data, 
            title=f"{title} - Train Loss",
            save_name=f"{experiment_name}_2d_train.png",
            use_test=False,
            dpi=self.config['visualization']['dpi']
        )
        
        visualizer.plot_2d_landscape(
            landscape_data,
            title=f"{title} - Test Loss",
            save_name=f"{experiment_name}_2d_test.png",
            use_test=True,
            dpi=self.config['visualization']['dpi']
        )
        
        visualizer.plot_3d_landscape(
            landscape_data,
            title=f"{title} - 3D Train Loss",
            save_name=f"{experiment_name}_3d_train.png",
            use_test=False,
            dpi=self.config['visualization']['dpi']
        )
        
        visualizer.plot_3d_landscape(
            landscape_data,
            title=f"{title} - 3D Test Loss",
            save_name=f"{experiment_name}_3d_test.png",
            use_test=True,
            dpi=self.config['visualization']['dpi']
        )
        
        visualizer.plot_training_curves(
            training_results['train_losses'],
            training_results['test_losses'],
            training_results['train_accs'],
            training_results['test_accs'],
            save_name=f"{experiment_name}_curves.png",
            dpi=self.config['visualization']['dpi']
        )
        print()
        
        # Compile results
        results = {
            'experiment': experiment_name,
            'model_name': model_name,
            'use_skip_connections': use_skip,
            'use_batchnorm': use_batchnorm,
            'dataset': dataset,
            'input_channels': num_input_channels,  # ADDED FOR REFERENCE
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'training_results': training_results,
            'geometric_metrics': {
                'sharpness': sharpness_metrics,
            },
            'landscape_stats': {
                'train': landscape_stats_train,
                'test': landscape_stats_test
            }
        }
        
        return results


if __name__ == '__main__':
    pipeline = LossLandscapePipeline('config.yaml')
    results = pipeline.run_all_experiments()
