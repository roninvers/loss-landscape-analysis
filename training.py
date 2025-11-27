# FILE: training.py
# Purpose: Training loop with metrics collection
# Copy this entire content into a file named 'training.py'

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from tqdm import tqdm
import numpy as np


class Trainer:
    """Training loop with comprehensive metrics collection."""
    
    def __init__(self, model, device='cpu', dtype=torch.float32):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.snapshots = {}
    
    def train_epoch(self, train_loader, optimizer):
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in progress_bar:
            images = images.to(self.device).to(self.dtype)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': total_loss / (total/images.size(0))})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device).to(self.dtype)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, test_loader, epochs=50, 
              learning_rate=0.1, optimizer_name='sgd',
              save_snapshots_at=[0.1, 0.5, 0.9]):
        """Full training loop with snapshots at key points."""
        
        if optimizer_name == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=learning_rate,
                           momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        print(f"Training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            test_loss, test_acc = self.evaluate(test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            
            scheduler.step()
            
            # Save snapshots at key training points
            progress = (epoch + 1) / epochs
            for snapshot_point in save_snapshots_at:
                if abs(progress - snapshot_point) < 0.02:
                    snapshot_key = f"{int(snapshot_point*100)}%"
                    if snapshot_key not in self.snapshots:
                        self.snapshots[snapshot_key] = {
                            'model_state': self._get_model_parameters(),
                            'epoch': epoch,
                            'train_acc': train_acc,
                            'test_acc': test_acc,
                            'train_loss': train_loss,
                            'test_loss': test_loss
                        }
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
                      f"Train Acc: {train_acc:6.2f}%, Test Acc: {test_acc:6.2f}%")
        
        return {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs,
            'final_test_acc': test_acc,
            'generalization_gap': self.train_accs[-1] - self.test_accs[-1]
        }
    
    def _get_model_parameters(self):
        """Extract model parameters as dictionary."""
        return {name: param.data.clone() for name, param in 
                self.model.named_parameters()}


if __name__ == '__main__':
    print("Training module loaded successfully!")
