import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.config import DROPOUT_RATE # Import DROPOUT_RATE from config

class LSTMBotDetector(nn.Module):
    """
    LSTM-based model for bot detection that combines text features with numerical and boolean features
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_numerical_features: int = 8,
        num_boolean_features: int = 1,
        num_classes: int = 2,
        dropout_prob: float = DROPOUT_RATE, # Use DROPOUT_RATE from config
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM-based bot detection model
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            num_numerical_features: Number of numerical features
            num_boolean_features: Number of boolean features
            num_classes: Number of output classes (default: 2 for binary classification)
            dropout_prob: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMBotDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # *** FIX: Simplified Attention Mechanism ***
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention_weights = nn.Linear(lstm_output_dim, 1)
        # -----------------------------------------
        
        # Feature processing layers
        self.numerical_layer = nn.Linear(num_numerical_features, 64)
        self.boolean_layer = nn.Linear(num_boolean_features, 16)
        
        # Combined features dimension
        self.combined_dim = lstm_output_dim + 64 + 16
        
        # Classification layers
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        modules_to_initialize = [
            self.numerical_layer, self.boolean_layer, 
            self.attention_weights, self.classifier
        ]
        for module in modules_to_initialize:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor, # Although passed, not explicitly used by standard LSTM/Attention
        numerical_features: torch.Tensor,
        boolean_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            numerical_features: Numerical features tensor [batch_size, num_numerical]
            boolean_features: Boolean features tensor [batch_size, num_boolean]
            
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Get embeddings [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        
        # Apply LSTM [batch_size, seq_len, hidden_dim * num_directions]
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # *** FIX: Apply Simplified Attention ***
        # Calculate attention scores [batch_size, seq_len, 1]
        attn_scores = torch.tanh(self.attention_weights(lstm_output))
        
        # Apply softmax to get weights [batch_size, seq_len, 1]
        # Use attention_mask to mask padding tokens before softmax
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9) # Mask padding
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Compute context vector [batch_size, hidden_dim * num_directions]
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        # ---------------------------------------
        
        # Process numerical features [batch_size, 64]
        numerical_output = torch.relu(self.numerical_layer(numerical_features))
        
        # Process boolean features [batch_size, 16]
        boolean_output = torch.relu(self.boolean_layer(boolean_features))
        
        # Combine all features [batch_size, combined_dim]
        combined_features = torch.cat([context_vector, numerical_output, boolean_output], dim=1)
        combined_features = self.dropout(combined_features)
        
        # Classification [batch_size, num_classes]
        logits = self.classifier(combined_features)
        
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
            input_ids: Token IDs
            attention_mask: Attention mask
            numerical_features: Numerical features tensor
            boolean_features: Boolean features tensor
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                boolean_features=boolean_features
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions, probabilities

