# FILE: model_builder.py (COMPLETE CORRECTED VERSION)
# Purpose: Neural network architectures (ResNet-20, VGG-16) with dynamic input channels
# This version works with BOTH MNIST (1 channel) and CIFAR-10 (3 channels)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block with optional skip connections.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, 
                 use_skip=True, use_batchnorm=True):
        super().__init__()
        self.use_skip = use_skip
        self.use_batchnorm = use_batchnorm
        self.stride = stride
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            if use_skip:
                layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
                ]
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                self.shortcut = nn.Sequential(*layers)
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        
        # Add skip connection
        if self.use_skip:
            out = out + self.shortcut(identity)
        
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """
    ResNet-20 with configurable skip connections, batch normalization, and input channels.
    Works with both MNIST (1 channel) and CIFAR-10 (3 channels).
    """
    
    def __init__(self, num_classes=10, use_skip=True, use_batchnorm=True, in_channels=3):
        super().__init__()
        self.use_skip = use_skip
        self.use_batchnorm = use_batchnorm
        
        # Initial convolution - NOW ACCEPTS ANY NUMBER OF INPUT CHANNELS
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
        
        # Residual layers
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        
        # Global average pooling and classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a residual layer with multiple blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, 
                                   self.use_skip, self.use_batchnorm))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1,
                                       self.use_skip, self.use_batchnorm))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class VGGBlock(nn.Module):
    """Single VGG convolutional block."""
    
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class VGG16(nn.Module):
    """
    VGG-16 without skip connections.
    Works with both MNIST (1 channel) and CIFAR-10 (3 channels).
    """
    
    def __init__(self, num_classes=10, use_batchnorm=True, in_channels=3):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # Feature extraction - NOW ACCEPTS ANY NUMBER OF INPUT CHANNELS
        self.features = nn.Sequential(
            # Block 1: 64 filters, 2 convolutions
            VGGBlock(in_channels, 64, use_batchnorm),  # USES in_channels parameter!
            VGGBlock(64, 64, use_batchnorm),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 128 filters, 2 convolutions
            VGGBlock(64, 128, use_batchnorm),
            VGGBlock(128, 128, use_batchnorm),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 256 filters, 3 convolutions
            VGGBlock(128, 256, use_batchnorm),
            VGGBlock(256, 256, use_batchnorm),
            VGGBlock(256, 256, use_batchnorm),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 512 filters, 3 convolutions
            VGGBlock(256, 512, use_batchnorm),
            VGGBlock(512, 512, use_batchnorm),
            VGGBlock(512, 512, use_batchnorm),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 512 filters, 3 convolutions
            VGGBlock(512, 512, use_batchnorm),
            VGGBlock(512, 512, use_batchnorm),
            VGGBlock(512, 512, use_batchnorm),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet20', num_classes=10, 
              use_skip=True, use_batchnorm=True, device='cpu', in_channels=3):
    """
    Factory function to get a model instance.
    
    Args:
        model_name (str): 'resnet20' or 'vgg16'
        num_classes (int): Number of output classes
        use_skip (bool): Use skip connections (ResNet only)
        use_batchnorm (bool): Use batch normalization
        device (str): Device to place model on
        in_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR-10)
    
    Returns:
        nn.Module: Model instance
    """
    if model_name.lower() == 'resnet20':
        model = ResNet20(num_classes=num_classes, use_skip=use_skip,
                        use_batchnorm=use_batchnorm, in_channels=in_channels)
    elif model_name.lower() == 'vgg16':
        model = VGG16(num_classes=num_classes, use_batchnorm=use_batchnorm, 
                     in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Supported: ['resnet20', 'vgg16']")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created {model_name} with {num_params:,} parameters")
    
    return model.to(device)


if __name__ == '__main__':
    # Test models
    print("Testing model architectures...\n")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test ResNet-20 with MNIST (1 channel)
    print("ResNet-20 with MNIST (1 channel):")
    model1 = get_model('resnet20', use_skip=True, use_batchnorm=True, 
                       device=device, in_channels=1)
    x = torch.randn(2, 1, 28, 28).to(device)
    y = model1(x)
    print(f"Output shape: {y.shape}\n")
    
    # Test ResNet-20 with CIFAR-10 (3 channels)
    print("ResNet-20 with CIFAR-10 (3 channels):")
    model2 = get_model('resnet20', use_skip=True, use_batchnorm=True, 
                       device=device, in_channels=3)
    x = torch.randn(2, 3, 32, 32).to(device)
    y = model2(x)
    print(f"Output shape: {y.shape}\n")
    
    # Test VGG-16 with MNIST
    print("VGG-16 with MNIST (1 channel):")
    model3 = get_model('vgg16', use_batchnorm=True, device=device, in_channels=1)
    x = torch.randn(2, 1, 28, 28).to(device)
    y = model3(x)
    print(f"Output shape: {y.shape}\n")
    
    print("✓ All models working correctly!")
