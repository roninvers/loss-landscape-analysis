

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import os


class LandscapeVisualizer:
    """Visualization of loss landscapes and training metrics."""
    
    def __init__(self, save_dir='./results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_2d_landscape(self, landscape_data, title='Loss Landscape',
                         save_name='landscape_2d.png', use_test=False, dpi=300):
        """Plot 2D contour plot of loss landscape."""
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        losses = landscape_data['losses_test'] if use_test else landscape_data['losses_train']
        X = landscape_data['X']
        Y = landscape_data['Y']
        
        # Create contour plot
        levels = np.linspace(losses.min(), losses.max(), 25)
        contour_fill = ax.contourf(X, Y, losses, levels=levels, cmap='viridis', alpha=0.9)
        contour_lines = ax.contour(X, Y, losses, levels=levels, colors='black', 
                                   alpha=0.2, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour_fill, ax=ax)
        cbar.set_label('Loss', rotation=270, labelpad=20, fontsize=11)
        
        # Labels and title
        ax.set_xlabel('Direction 1 (α)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Direction 2 (β)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved 2D landscape to {self.save_dir}/{save_name}")
        plt.close()
    
    def plot_3d_landscape(self, landscape_data, title='3D Loss Landscape',
                         save_name='landscape_3d.png', use_test=False, dpi=300):
        """Plot 3D surface plot of loss landscape."""
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        losses = landscape_data['losses_test'] if use_test else landscape_data['losses_train']
        X = landscape_data['X']
        Y = landscape_data['Y']
        
        # Surface plot
        surf = ax.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')
        
        # Labels and title
        ax.set_xlabel('Direction 1 (α)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Direction 2 (β)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Adjust viewing angle
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved 3D landscape to {self.save_dir}/{save_name}")
        plt.close()
    
    def plot_training_curves(self, train_losses, test_losses, 
                            train_accs, test_accs,
                            save_name='training_curves.png', dpi=300):
        """Plot training and validation curves."""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = np.arange(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2.5, marker='o', markersize=3)
        axes[0].plot(epochs, test_losses, label='Test Loss', linewidth=2.5, marker='s', markersize=3)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Test Loss', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, train_accs, label='Train Accuracy', linewidth=2.5, marker='o', markersize=3)
        axes[1].plot(epochs, test_accs, label='Test Accuracy', linewidth=2.5, marker='s', markersize=3)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training and Test Accuracy', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11, loc='lower right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved training curves to {self.save_dir}/{save_name}")
        plt.close()
    
    def plot_comparison(self, results_dict, metric='final_test_acc',
                       save_name='comparison.png', dpi=300):
        """Compare results across architectures."""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = list(results_dict.keys())
        values = [results_dict[n]['training_results'][metric] 
                 if metric in results_dict[n]['training_results']
                 else results_dict[n]['geometric_metrics'][metric]
                 for n in names]
        
        # Create bars with gradient colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison: {metric}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved comparison to {self.save_dir}/{save_name}")
        plt.close()
    
    def plot_metrics_heatmap(self, results_dict, save_name='metrics_heatmap.png', dpi=300):
        """Create heatmap of key metrics across architectures."""
        
        metrics_data = []
        model_names = []
        
        for key, result in results_dict.items():
            model_names.append(key.replace('_', '\n'))
            metrics_data.append([
                result['training_results']['final_test_acc'],
                result['training_results']['generalization_gap'],
                result['geometric_metrics']['hessian']['max_eigenvalue'],
                result['geometric_metrics']['hessian']['condition_number']
            ])
        
        metrics_array = np.array(metrics_data).T
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(metrics_array, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(model_names)
        ax.set_yticklabels(['Test Acc (%)', 'Gen Gap (%)', 'Max Eigenvalue', 'Condition Number'])
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(4):
            for j in range(len(model_names)):
                text = ax.text(j, i, f'{metrics_array[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title("Metrics Comparison Across Architectures", fontsize=13, fontweight='bold', pad=15)
        fig.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved metrics heatmap to {self.save_dir}/{save_name}")
        plt.close()


if __name__ == '__main__':
    print("Visualization module loaded successfully!")
