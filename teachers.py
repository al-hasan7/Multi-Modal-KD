import torch
import torch.nn as nn
from torchvision.models import resnet18

class ChannelAdapter(nn.Module):
    """
    Learnable channel projection:
      - 1 channel → 3 channels (Conv1x1)
      - 3 channels → identity
    """
    def __init__(self):
        super().__init__()
        self.gray_to_rgb = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, bias=False)

    def forward(self, x):
        if x.shape[1] == 1:
            return self.gray_to_rgb(x)
        elif x.shape[1] == 3:
            return x
        else:
            raise ValueError(
                f"Expected 1 or 3 channels, got {x.shape[1]}"
            )


class MultiDomainResNet18(nn.Module):
    """
    ResNet18 with Domain-Specific BatchNorm and multi-head classification
    """
    def __init__(self, num_classes=32, use_dsbn=True, num_domains=3, 
                 extract_features=False):
        super().__init__()
        self.use_dsbn = use_dsbn
        self.num_domains = num_domains
        self.extract_features = extract_features
        self.channel_adapter = ChannelAdapter()
        
        # Import ResNet18 backbone
        from torchvision.models import resnet18
        base_model = resnet18(pretrained=False)
        
        # first conv (stem) for 28x28 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if use_dsbn:
            # Replace all BatchNorm layers with DSBN
            self.bn1 = DomainSpecificBatchNorm2d(64, num_domains)
        else:
            self.bn1 = base_model.bn1
        
        self.relu = base_model.relu
        self.maxpool = nn.Identity()  # Remove maxpool for small images
        
        # Copy layer blocks
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # If using DSBN, replace BN in all residual blocks
        if use_dsbn:
            self._replace_bn_with_dsbn(self.layer1)
            self._replace_bn_with_dsbn(self.layer2)
            self._replace_bn_with_dsbn(self.layer3)
            self._replace_bn_with_dsbn(self.layer4)
        
        self.avgpool = base_model.avgpool
        
        # Feature dimension
        self.feature_dim = 512
        
        # Classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
    def _replace_bn_with_dsbn(self, layer):
        """Recursively replace BatchNorm2d with DomainSpecificBatchNorm2d"""
        for name, module in layer.named_children():
            if isinstance(module, nn.BatchNorm2d):
                dsbn = DomainSpecificBatchNorm2d(
                    module.num_features, 
                    self.num_domains,
                    module.eps,
                    module.momentum
                )
                setattr(layer, name, dsbn)
            else:
                self._replace_bn_with_dsbn(module)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[1] in (1, 3):
                return x
            elif x.shape[-1] in (1, 3):
                return x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            if x.shape[0] in (1, 3):
                return x.unsqueeze(0)
            elif x.shape[-1] in (1, 3):
                return x.permute(2, 0, 1).unsqueeze(0)

        raise ValueError(f"Unsupported input shape: {x.shape}")

    
    def _forward_with_dsbn(self, x, domain_id):
        """Forward pass using domain-specific batch norm"""
        x = self.conv1(x)
        x = self.bn1(x, domain_id)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self._forward_layer_with_dsbn(self.layer1, x, domain_id)
        x = self._forward_layer_with_dsbn(self.layer2, x, domain_id)
        x = self._forward_layer_with_dsbn(self.layer3, x, domain_id)
        x = self._forward_layer_with_dsbn(self.layer4, x, domain_id)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        
        if self.extract_features:
            return logits, features
        return logits
    
    def _forward_layer_with_dsbn(self, layer, x, domain_id):
        """Forward through a layer with DSBN support"""
        for block in layer:
            identity = x
            
            # Forward through block with DSBN
            out = block.conv1(x)
            out = block.bn1(out, domain_id) if isinstance(block.bn1, DomainSpecificBatchNorm2d) else block.bn1(out)
            out = block.relu(out)
            
            out = block.conv2(out)
            out = block.bn2(out, domain_id) if isinstance(block.bn2, DomainSpecificBatchNorm2d) else block.bn2(out)
            
            if block.downsample is not None:
                # Handle downsample with DSBN
                identity = block.downsample[0](x)  # Conv
                if isinstance(block.downsample[1], DomainSpecificBatchNorm2d):
                    identity = block.downsample[1](identity, domain_id)
                else:
                    identity = block.downsample[1](identity)
            
            out += identity
            out = block.relu(out)
            x = out
        
        return x
    
    def forward(self, x, domain_id=None):
        x = self._to_nchw(x)  # normalize input layout to NCHW
        x = self.channel_adapter(x)

        if self.use_dsbn:
            if domain_id is None:
                raise ValueError("domain_id required when using DSBN")
            return self._forward_with_dsbn(x, domain_id)
        else:
            # Standard forward without DSBN
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            features = torch.flatten(x, 1)
            logits = self.fc(features)
            
            if self.extract_features:
                return logits, features
            return logits


