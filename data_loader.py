

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import os


class DataLoaderManager:
    """
    Manages dataset loading with reproducibility and preprocessing.
    
    Supports CIFAR-10 and MNIST with configurable augmentation.
    """
    
    def __init__(self, dataset_name='cifar10', batch_size=32, 
                 num_workers=0, pin_memory=False, augmentation=True):
        """
        Initialize DataLoaderManager.
        
        Args:
            dataset_name (str): 'cifar10' or 'mnist'
            batch_size (int): Batch size for DataLoader
            num_workers (int): Number of workers (0 for MPS)
            pin_memory (bool): Pin memory (False for MPS)
            augmentation (bool): Use data augmentation
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augmentation = augmentation
        
        # Create data directory if not exists
        os.makedirs('./data', exist_ok=True)
    
    def get_transforms(self):
        """
        Get dataset-specific transforms.
        
        Returns:
            tuple: (train_transform, test_transform)
        """
        if self.dataset_name == 'cifar10':
            if self.augmentation:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)
                    )
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010)
                    )
                ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ])
            
        elif self.dataset_name == 'mnist':
            if self.augmentation:
                train_transform = transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return train_transform, test_transform
    
    def load_dataset(self):
        """
        Load training and test datasets.
        
        Returns:
            tuple: (train_set, test_set)
        """
        train_transform, test_transform = self.get_transforms()
        
        if self.dataset_name == 'cifar10':
            train_set = datasets.CIFAR10(
                root='./data', 
                train=True, 
                download=True, 
                transform=train_transform
            )
            test_set = datasets.CIFAR10(
                root='./data', 
                train=False,
                download=True, 
                transform=test_transform
            )
            
        elif self.dataset_name == 'mnist':
            train_set = datasets.MNIST(
                root='./data', 
                train=True,
                download=True, 
                transform=train_transform
            )
            test_set = datasets.MNIST(
                root='./data', 
                train=False,
                download=True, 
                transform=test_transform
            )
        
        return train_set, test_set
    
    def get_loaders(self):
        """
        Get PyTorch DataLoaders for train and test sets.
        
        Returns:
            tuple: (train_loader, test_loader, train_size, test_size)
        """
        train_set, test_set = self.load_dataset()
        
        train_loader = DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_set, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, len(train_set), len(test_set)
    
    def get_sample_batch(self):
        """
        Get a single batch for inspection.
        
        Returns:
            tuple: (images, labels)
        """
        train_loader, _, _, _ = self.get_loaders()
        images, labels = next(iter(train_loader))
        return images, labels


if __name__ == '__main__':
    # Test the data loader
    print("Testing DataLoaderManager...")
    
    manager = DataLoaderManager(dataset_name='cifar10', batch_size=32)
    train_loader, test_loader, train_size, test_size = manager.get_loaders()
    
    print(f"Dataset: CIFAR-10")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {test_size}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print("✓ DataLoaderManager working correctly!")
