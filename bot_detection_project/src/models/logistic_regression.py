import torch
import torch.nn as nn
from typing import Tuple

class LogisticRegressionBotDetector(nn.Module):
    """
    Logistic Regression baseline model implemented using PyTorch nn.Linear
    Accepts combined features (BERT output + numerical + boolean)
    """
    def __init__(
        self,
        input_dim: int, # Combined dimension from BERT + numerical + boolean features
        num_classes: int = 2
    ):
        """
        Initialize the Logistic Regression model
        
        Args:
            input_dim: Total dimension of the combined input features
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super(LogisticRegressionBotDetector, self).__init__()
        
        # Single linear layer for logistic regression
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            combined_features: Combined tensor of BERT output, numerical, and boolean features
            
        Returns:
            Output logits
        """
        # Apply the linear layer
        logits = self.linear(combined_features)
        return logits

    # Note: This model expects the features to be pre-combined before being passed to forward.
    # The trainer or pipeline script will need to handle the feature combination logic
    # similar to how it's done within the BERT/LSTM models if this LR model is used.
    # For simplicity in swapping, we might adjust the BERT/LSTM models to also accept
    # pre-combined features or adjust the trainer to handle this difference.
    # For now, we assume the trainer will adapt. A more robust implementation might involve
    # creating a base model class or a wrapper.

    def predict(self, combined_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the model
        
        Args:
            combined_features: Combined tensor of features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(combined_features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions, probabilities
