import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    """
    Entropy Loss for encouraging uncertainty/diversity in predictions.
    
    For binary classification, maximum entropy occurs when p(class=1) = 0.5.
    H(p) = -p*log(p) - (1-p)*log(1-p)
    
    This loss is useful for actual spoof samples to encourage the model
    to learn diverse representations at the high level.
    """
    
    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'
        """
        super(EntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits, eps=1e-8):
        """
        Compute entropy loss from logits.
        
        Args:
            logits (torch.Tensor): Logits from model, shape [N, num_classes]
            eps (float): Small constant for numerical stability
            
        Returns:
            torch.Tensor: Entropy loss (to be maximized, so return negative)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate entropy: H = -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        
        # Return negative entropy (so minimizing loss = maximizing entropy)
        if self.reduction == 'mean':
            return -entropy.mean()
        elif self.reduction == 'sum':
            return -entropy.sum()
        else:  # 'none'
            return -entropy


class BinaryEntropyLoss(nn.Module):
    """
    Binary Entropy Loss specifically for binary classification.
    Encourages predictions toward maximum uncertainty (p=0.5).
    """
    
    def __init__(self, reduction='mean'):
        super(BinaryEntropyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits, eps=1e-8):
        """
        Compute binary entropy from logits for binary classification.
        
        Args:
            logits (torch.Tensor): Logits for binary classification, shape [N, 2]
            eps (float): Small constant for numerical stability
            
        Returns:
            torch.Tensor: Negative entropy (minimizing = maximizing entropy)
        """
        # Get probability of class 1 (spoof)
        probs = F.softmax(logits, dim=1)
        p = probs[:, 1]  # Probability of spoof class
        
        # Clamp to avoid log(0)
        p = torch.clamp(p, eps, 1 - eps)
        
        # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        
        # Return negative (to maximize entropy via minimizing loss)
        if self.reduction == 'mean':
            return -entropy.mean()
        elif self.reduction == 'sum':
            return -entropy.sum()
        else:  # 'none'
            return -entropy
