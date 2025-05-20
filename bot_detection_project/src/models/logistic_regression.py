import torch
import torch.nn as nn
from typing import Tuple

class LogisticRegressionBotDetector(nn.Module):
    """
    A Logistic Regression model that includes its own embedding layer for text and processes numerical/boolean features.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_numerical_features: int,
        num_boolean_features: int,
        num_classes: int = 2
    ):
        """
        Initialize the Logistic Regression model
        
        Args:
            vocab_size: Size of the vocabulary for text features
            embedding_dim: Dimension of the text embeddings
            num_numerical_features: Number of numerical features
            num_boolean_features: Number of boolean features
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super(LogisticRegressionBotDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Single linear layer for logistic regression
        # Input dimension for the linear layer: embedding_dim + num_numerical_features + num_boolean_features
        self.linear = nn.Linear(embedding_dim + num_numerical_features + num_boolean_features, num_classes)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        boolean_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            input_ids: Tensor of input IDs for text [batch_size, seq_len]
            attention_mask: Tensor of attention masks for text [batch_size, seq_len]
            numerical_features: Tensor of numerical features [batch_size, num_numerical_features]
            boolean_features: Tensor of boolean features [batch_size, num_boolean_features]
            
        Returns:
            Output logits
        """
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Mask out padding tokens
        masked_embedded = embedded * attention_mask.unsqueeze(-1).float()
        
        # Sum embeddings along sequence length
        summed_embeddings = masked_embedded.sum(dim=1)  # [batch_size, embedding_dim]
        
        # Count non-masked tokens
        non_masked_count = attention_mask.sum(dim=1).unsqueeze(-1).float() # [batch_size, 1]
        
        # Calculate mean pooled embeddings (clamp to avoid division by zero)
        # Aggregate token embeddings using attention-masked mean pooling
        mean_embeddings = summed_embeddings / torch.clamp(non_masked_count, min=1e-9)
        
        # Concatenate features
        combined_features = torch.cat([mean_embeddings, numerical_features, boolean_features], dim=1)
        
        # Apply the linear layer
        logits = self.linear(combined_features)
        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        boolean_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the model
        
        Args:
            input_ids: Tensor of input IDs for text
            attention_mask: Tensor of attention masks for text
            numerical_features: Tensor of numerical features
            boolean_features: Tensor of boolean features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, numerical_features, boolean_features)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions, probabilities
