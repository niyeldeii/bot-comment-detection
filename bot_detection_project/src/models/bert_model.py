import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTBotDetector(nn.Module):
    """
    Optimized RoBERTa-large model for cross-platform bot detection
    """
    
    def __init__(
        self,
        num_numerical_features: int,
        num_boolean_features: int,
        bert_model_name: str = "roberta-large",
        dropout_prob: float = 0.3,
        hidden_size: int = 128  # Optimized hidden size for the dataset complexity
    ):
        """
        Initialize the model
        """
        super(BERTBotDetector, self).__init__()
        
        # Load BERT model and config
        self.config = AutoConfig.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name, config=self.config)
        
        # Get BERT output dimension
        bert_output_dim = self.config.hidden_size  # Usually 768 for bert-base
        
        # Simplified BERT processing
        self.bert_dropout = nn.Dropout(dropout_prob)
        # Simplified single-layer projection for dataset-specific optimization
        self.bert_linear = nn.Linear(bert_output_dim, hidden_size)
        self.bert_norm = nn.LayerNorm(hidden_size)
        
        # Simplified numerical features processing
        self.numerical_linear = nn.Linear(num_numerical_features, hidden_size)
        self.numerical_norm = nn.LayerNorm(hidden_size)
        
        # Simplified boolean features processing
        self.boolean_linear = nn.Linear(num_boolean_features, hidden_size)
        self.boolean_norm = nn.LayerNorm(hidden_size)
        
        # Final classifier
        combined_dim = hidden_size * 3  # BERT + numerical + boolean
        # Simplified classifier for dataset-specific optimization
        self.classifier_dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(combined_dim, 2)  # Binary classification
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize weights for better convergence
        """
        # Initialize linear layers
        nn.init.xavier_uniform_(self.bert_linear.weight)
        nn.init.zeros_(self.bert_linear.bias)
        
        nn.init.xavier_uniform_(self.numerical_linear.weight)
        nn.init.zeros_(self.numerical_linear.bias)
        
        nn.init.xavier_uniform_(self.boolean_linear.weight)
        nn.init.zeros_(self.boolean_linear.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        boolean_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with simplified architecture optimized for the dataset
        """
        # Process text with BERT
        # token_type_ids are omitted as they are not typically used by RoBERTa for single-sequence inputs.
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Get CLS token and process
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        cls_hidden_state = self.bert_dropout(cls_hidden_state)
        # Simplified single-layer projection
        bert_features = self.bert_linear(cls_hidden_state)
        bert_features = self.bert_norm(bert_features)
        bert_features = torch.relu(bert_features)
        
        # Process numerical features
        numerical_features = self.numerical_linear(numerical_features)
        numerical_features = self.numerical_norm(numerical_features)
        numerical_features = torch.relu(numerical_features)
        
        # Process boolean features
        boolean_features = self.boolean_linear(boolean_features)
        boolean_features = self.boolean_norm(boolean_features)
        boolean_features = torch.relu(boolean_features)
        
        # Combine features
        combined_features = torch.cat([bert_features, numerical_features, boolean_features], dim=1)
        
        # Classification
        # Simplified classifier
        combined_features = self.classifier_dropout(combined_features)
        logits = self.classifier(combined_features)
        
        return logits
