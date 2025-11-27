

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class LandscapeAnalyzer:
    """Core loss landscape probing and analysis."""
    
    def __init__(self, model, train_loader, test_loader, device='cpu', dtype=torch.float32):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dtype = dtype
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, data_loader=None, use_train=True):
        """Compute average loss on dataset."""
        if data_loader is None:
            data_loader = self.train_loader if use_train else self.test_loader
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device).to(self.dtype)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def get_random_directions(self, num_directions=2):
        """Generate random direction vectors in parameter space."""
        params = list(self.model.parameters())
        directions = []
        
        for _ in range(num_directions):
            direction = []
            total_params = sum(p.numel() for p in params)
            for param in params:
                d = torch.randn_like(param) / (np.sqrt(total_params))
                direction.append(d)
            directions.append(direction)
        
        return directions
    
    def perturb_model(self, alpha_values, beta_values, 
                     direction1, direction2):
        """Perturb model along two directions and compute losses."""
        
        losses_train = np.zeros((len(alpha_values), len(beta_values)))
        losses_test = np.zeros_like(losses_train)
        
        # Store original parameters
        original_params = [param.data.clone() for param in self.model.parameters()]
        
        progress = tqdm(total=len(alpha_values) * len(beta_values),
                       desc="Computing landscape", leave=False)
        
        for i, alpha in enumerate(alpha_values):
            for j, beta in enumerate(beta_values):
                # Apply perturbation
                for param, orig, d1, d2 in zip(self.model.parameters(),
                                              original_params, direction1, direction2):
                    param.data = orig + alpha * d1 + beta * d2
                
                # Compute losses
                loss_train = self.compute_loss(use_train=True)
                loss_test = self.compute_loss(use_train=False)
                
                losses_train[i, j] = loss_train
                losses_test[i, j] = loss_test
                progress.update(1)
        
        # Restore original parameters
        for param, orig in zip(self.model.parameters(), original_params):
            param.data = orig
        
        progress.close()
        return losses_train, losses_test
    
    def compute_2d_landscape(self, alpha_range=(-1, 1), 
                            beta_range=(-1, 1), num_samples=50):
        """Compute 2D loss landscape using random directions."""
        print("Generating random directions...")
        directions = self.get_random_directions(num_directions=2)
        
        alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_samples)
        beta_values = np.linspace(beta_range[0], beta_range[1], num_samples)
        
        print("Computing 2D landscape...")
        losses_train, losses_test = self.perturb_model(
            alpha_values, beta_values, directions[0], directions[1]
        )
        
        # Create mesh grid
        X, Y = np.meshgrid(alpha_values, beta_values)
        
        return {
            'alpha_values': alpha_values,
            'beta_values': beta_values,
            'losses_train': losses_train,
            'losses_test': losses_test,
            'X': X,
            'Y': Y,
            'directions': directions
        }
    
    def get_landscape_statistics(self, landscape_data, data_type='train'):
        """Compute statistics of landscape."""
        losses = landscape_data['losses_train'] if data_type == 'train' else landscape_data['losses_test']
        
        return {
            'min': float(losses.min()),
            'max': float(losses.max()),
            'mean': float(losses.mean()),
            'std': float(losses.std()),
            'range': float(losses.max() - losses.min()),
        }


if __name__ == '__main__':
    print("LandscapeAnalyzer module loaded successfully!")
