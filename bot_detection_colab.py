import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

# Add a Colab-specific setup section
COLAB_SETUP = """
# Mount Google Drive to access the data
from google.colab import drive
drive.mount("/content/drive")

# Install required packages
!pip install pandas torch transformers scikit-learn imbalanced-learn joblib

# Unzip the project if needed
# !unzip /content/drive/MyDrive/path_to_your_zip/bot_detection_project_fixed.zip -d /content/
"""

class TwitterDataset(Dataset):
    """PyTorch Dataset for Twitter data"""
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: BertTokenizer,
        scaler: StandardScaler,
        max_len: int = 128,
        is_test: bool = False,
        text_feature: str = "Tweet",
        target_feature: str = "Bot Label",
        numerical_features: List[str] = None,
        boolean_features: List[str] = None
    ):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.max_len = max_len
        self.is_test = is_test
        self.text_feature = text_feature
        self.target_feature = target_feature
        
        # Default features if none provided
        if numerical_features is None:
            self.numerical_features = ["Retweet Count", "Mention Count", "Follower Count", 
                                      "hour", "day_of_week", "month", "year"]
        else:
            self.numerical_features = numerical_features
            
        if boolean_features is None:
            self.boolean_features = ["Verified"]
        else:
            self.boolean_features = boolean_features
        
        # Ensure columns exist
        missing_num_cols = [col for col in self.numerical_features if col not in self.dataframe.columns]
        if missing_num_cols:
            raise ValueError(f"Missing numerical columns in dataframe: {missing_num_cols}")
            
        missing_bool_cols = [col for col in self.boolean_features if col not in self.dataframe.columns]
        if missing_bool_cols:
            raise ValueError(f"Missing boolean columns in dataframe: {missing_bool_cols}")
        
        # Scale numerical features
        self.numerical_data = self.scaler.transform(self.dataframe[self.numerical_features])
        
        # Convert boolean features to integers
        self.boolean_data = self.dataframe[self.boolean_features].astype(int).values
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.dataframe.iloc[idx]["cleaned_tweet"] if "cleaned_tweet" in self.dataframe.columns else self.dataframe.iloc[idx][self.text_feature]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Get numerical and boolean features
        numerical_features = torch.FloatTensor(self.numerical_data[idx])
        boolean_features = torch.FloatTensor(self.boolean_data[idx])
        
        # Combine all features
        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "numerical_features": numerical_features,
            "boolean_features": boolean_features
        }
        
        if not self.is_test and self.target_feature in self.dataframe.columns:
            sample["targets"] = torch.tensor(self.dataframe.iloc[idx][self.target_feature], dtype=torch.long)
            
        return sample

class BERTBotDetector(nn.Module):
    """
    BERT-based model for bot detection with enhanced feature integration.
    Uses BERT pooler output, improved feature combination, and a more robust architecture.
    """
    def __init__(
        self,
        num_numerical_features: int,
        num_boolean_features: int,
        num_classes: int = 2,
        dropout_prob: float = 0.3,
        model_name: str = "bert-base-uncased"
    ):
        super(BERTBotDetector, self).__init__()
        
        # Load pre-trained BERT model
        from transformers import BertModel, BertConfig
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_config = BertConfig.from_pretrained(model_name)
        
        # Dimensions
        self.bert_dim = self.bert_config.hidden_size
        self.num_numerical_features = num_numerical_features
        self.num_boolean_features = num_boolean_features
        
        # Process numerical features separately before combining
        self.numerical_processor = nn.Sequential(
            nn.Linear(self.num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Process boolean features separately before combining
        self.boolean_processor = nn.Sequential(
            nn.Linear(self.num_boolean_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Process BERT output separately
        self.bert_processor = nn.Sequential(
            nn.Linear(self.bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Combined features dimension after separate processing
        self.combined_processed_dim = 256 + 64 + 16
        
        # Layer Normalization after concatenation
        self.layer_norm = nn.LayerNorm(self.combined_processed_dim)
        
        # Intermediate layers with residual connections
        self.fc1 = nn.Linear(self.combined_processed_dim, 128)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Final classification layer
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize weights properly for better gradient flow
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        # List all modules/layers that need custom initialization
        modules_to_initialize = [
            self.numerical_processor, self.boolean_processor, 
            self.bert_processor, self.fc1, self.fc2, self.classifier
        ]
        
        for module in modules_to_initialize:
            if isinstance(module, nn.Sequential):
                # If it's a Sequential module, iterate through its layers
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(module, nn.Linear):
                # If it's a single Linear layer, initialize it directly
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # Add other layer types here if needed (e.g., nn.Conv2d)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        boolean_features: torch.Tensor
    ) -> torch.Tensor:
        # Process text with BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Process each feature type separately
        bert_features = self.bert_processor(bert_output.pooler_output)
        num_features = self.numerical_processor(numerical_features)
        bool_features = self.boolean_processor(boolean_features)
        
        # Combine processed features
        combined_features = torch.cat([bert_features, num_features, bool_features], dim=1)
        
        # Apply Layer Normalization
        normalized_features = self.layer_norm(combined_features)
        
        # First dense layer with ReLU and dropout
        x = self.fc1(normalized_features)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        
        # Second dense layer with ReLU and dropout
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits

def train_model_colab():
    """
    Complete training function for Google Colab
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    import re
    import datetime
    import os
    import time
    import joblib
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    # Configuration
    DATA_PATH = "/content/bot_detection_project/data/labeled_twitter_data.csv"
    MODEL_NAME = "bert-base-uncased"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    BERT_LR = 5e-5
    WARMUP_STEPS = 300
    RANDOM_SEED = 123
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 0.005
    GRAD_CLIP = 1.5
    
    # Features
    TEXT_FEATURE = "Tweet"
    NUMERICAL_FEATURES = ["Retweet Count", "Mention Count", "Follower Count"]
    BOOLEAN_FEATURES = ["Verified"]
    TARGET_FEATURE = "Bot Label"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    def set_seed(seed=RANDOM_SEED):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed()
    
    # Create directories
    os.makedirs("/content/models", exist_ok=True)
    os.makedirs("/content/results", exist_ok=True)
    
    # Data preprocessing functions
    def preprocess_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def extract_time_features(created_at):
        try:
            dt = datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            return {
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "month": dt.month,
                "year": dt.year
            }
        except:
            return {
                "hour": 0,
                "day_of_week": 0,
                "month": 0,
                "year": 0
            }
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean text data
    df["cleaned_tweet"] = df[TEXT_FEATURE].apply(preprocess_text)
    
    # Handle missing values in hashtags
    df["Hashtags"] = df["Hashtags"].fillna("")
    
    # Extract time features
    time_features = df["Created At"].apply(extract_time_features)
    time_df = pd.DataFrame(time_features.tolist())
    df = pd.concat([df, time_df], axis=1)
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=RANDOM_SEED,
        stratify=df[TARGET_FEATURE]
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15 / (1 - 0.15),
        random_state=RANDOM_SEED,
        stratify=train_val_df[TARGET_FEATURE]
    )
    
    print(f"Data loaded. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Fit scaler on training data
    numerical_cols_to_scale = NUMERICAL_FEATURES + ["hour", "day_of_week", "month", "year"]
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_cols_to_scale])
    
    # Save scaler
    joblib.dump(scaler, "/content/models/scaler.joblib")
    
    # Calculate class weights
    class_counts = train_df[TARGET_FEATURE].value_counts().to_dict()
    total = sum(class_counts.values())
    weights = {cls: total / count for cls, count in class_counts.items()}
    
    # Normalize weights
    weight_sum = sum(weights.values())
    weights = {cls: weight / weight_sum * len(weights) for cls, weight in weights.items()}
    
    # Convert to tensor
    class_weights = torch.FloatTensor([weights[i] for i in sorted(weights.keys())])
    print(f"Class weights: {class_weights}")
    
    # Create datasets and dataloaders
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = TwitterDataset(
        train_df, tokenizer, scaler, max_len=MAX_LEN,
        numerical_features=numerical_cols_to_scale
    )
    
    val_dataset = TwitterDataset(
        val_df, tokenizer, scaler, max_len=MAX_LEN,
        numerical_features=numerical_cols_to_scale
    )
    
    test_dataset = TwitterDataset(
        test_df, tokenizer, scaler, max_len=MAX_LEN, is_test=False,
        numerical_features=numerical_cols_to_scale
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = BERTBotDetector(
        num_numerical_features=len(numerical_cols_to_scale),
        num_boolean_features=len(BOOLEAN_FEATURES),
        dropout_prob=DROPOUT_RATE,
        model_name=MODEL_NAME
    )
    
    model = model.to(device)
    
    # Early stopping class
    class EarlyStopping:
        def __init__(self, patience=7, verbose=True, delta=0.001, path="/content/models/best_model.pth"):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float("inf")
            self.delta = delta
            self.path = path
            
        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
                
        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
    
    # Setup optimizer with parameter groups
    bert_params = {"params": [], "lr": BERT_LR, "weight_decay": WEIGHT_DECAY}
    numerical_processor_params = {"params": [], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    boolean_processor_params = {"params": [], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    bert_processor_params = {"params": [], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    classifier_params = {"params": [], "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    
    # Assign parameters to their respective groups
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith("bert."):
                bert_params["params"].append(param)
            elif name.startswith("numerical_processor."):
                numerical_processor_params["params"].append(param)
            elif name.startswith("boolean_processor."):
                boolean_processor_params["params"].append(param)
            elif name.startswith("bert_processor."):
                bert_processor_params["params"].append(param)
            else:
                classifier_params["params"].append(param)
    
    optimizer_grouped_parameters = [
        bert_params,
        numerical_processor_params,
        boolean_processor_params,
        bert_processor_params,
        classifier_params
    ]
    
    # Filter out empty parameter groups
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group["params"]) > 0]
    
    # Initialize optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    # Setup loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Setup scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True)
    
    # Training function
    def train_epoch():
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # Add gradient accumulation steps
        accumulation_steps = 2
        
        for batch_num, batch in enumerate(train_loader):
            # Zero gradients only at the beginning of accumulation steps
            if batch_num % accumulation_steps == 0:
                optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            boolean_features = batch["boolean_features"].to(device)
            targets = batch["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                boolean_features=boolean_features
            )
            
            loss = criterion(outputs, targets)
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            total_loss += loss.item() * accumulation_steps
            
            loss.backward()
            
            # Step optimizer and scheduler only at the end of accumulation steps
            if (batch_num + 1) % accumulation_steps == 0 or (batch_num + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
            if (batch_num + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Batch {batch_num + 1}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {current_lr:.2e} | Time: {elapsed_time:.2f}s")
                start_time = time.time()
                
        avg_train_loss = total_loss / len(train_loader)
        return avg_train_loss
    
    # Evaluation function
    def evaluate():
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                numerical_features = batch["numerical_features"].to(device)
                boolean_features = batch["boolean_features"].to(device)
                targets = batch["targets"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    boolean_features=boolean_features
                )
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                if probs.shape[1] > 1:
                    all_probs.extend(probs[:, 1].cpu().numpy())
                else:
                    all_probs.extend(probs.cpu().numpy().flatten())
                
        avg_val_loss = total_loss / len(val_loader)
        
        # Calculate metrics with improved error handling
        try:
            precision = precision_score(all_targets, all_preds, zero_division=0)
        except Exception as e:
            print(f"Warning: Precision calculation failed: {e}")
            precision = 0.0
            
        try:
            recall = recall_score(all_targets, all_preds, zero_division=0)
        except Exception as e:
            print(f"Warning: Recall calculation failed: {e}")
            recall = 0.0
            
        try:
            f1 = f1_score(all_targets, all_preds, zero_division=0)
        except Exception as e:
            print(f"Warning: F1 score calculation failed: {e}")
            f1 = 0.0
            
        try:
            if len(np.unique(all_targets)) > 1 and len(all_probs) == len(all_targets):
                 roc_auc = roc_auc_score(all_targets, all_probs)
            else:
                 roc_auc = 0.5
                 print("Warning: ROC AUC could not be calculated. Check target distribution and probability outputs.")
        except ValueError as e:
             roc_auc = 0.5
             print(f"Warning: ROC AUC calculation failed: {e}")

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        
        return avg_val_loss, metrics
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []}
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        start_epoch_time = time.time()
        
        avg_train_loss = train_epoch()
        history["train_loss"].append(avg_train_loss)
        
        avg_val_loss, metrics = evaluate()
        history["val_loss"].append(avg_val_loss)
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["f1_score"].append(metrics["f1_score"])
        history["roc_auc"].append(metrics["roc_auc"])
        
        epoch_duration = time.time() - start_epoch_time
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Precision: {metrics["precision"]:.4f}")
        print(f"  Recall: {metrics["recall"]:.4f}")
        print(f"  F1 Score: {metrics["f1_score"]:.4f}")
        print(f"  ROC AUC: {metrics["roc_auc"]:.4f}")
        print(f"  Duration: {epoch_duration:.2f}s")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"/content/models/model_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "metrics": metrics
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
                
    print("\nTraining finished.")
    
    # Load best model
    if os.path.exists("/content/models/best_model.pth"):
        print("Loading best model...")
        model.load_state_dict(torch.load("/content/models/best_model.pth", map_location=device))
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numerical_features = batch["numerical_features"].to(device)
            boolean_features = batch["boolean_features"].to(device)
            targets = batch["targets"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                boolean_features=boolean_features
            )
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
            if probs.shape[1] > 1:
                test_probs.extend(probs[:, 1].cpu().numpy())
            else:
                test_probs.extend(probs.cpu().numpy().flatten())
    
    # Calculate test metrics
    test_precision = precision_score(test_targets, test_preds, zero_division=0)
    test_recall = recall_score(test_targets, test_preds, zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_roc_auc = roc_auc_score(test_targets, test_probs)
    
    print("\nTest Results:")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  ROC AUC: {test_roc_auc:.4f}")
    
    # Save test results
    test_results = {
        "precision": test_precision,
        "recall": test_recall,
        "f1_score": test_f1,
        "roc_auc": test_roc_auc
    }
    
    import json
    with open("/content/results/test_results.json", "w") as f:
        json.dump(test_results, f)
    
    print("Test results saved to /content/results/test_results.json")
    
    return history, test_results

# Colab notebook code
COLAB_NOTEBOOK = """
# Bot Detection Model Training

This notebook runs the fixed bot detection model with all the implemented improvements.

## Setup and Installation

First, let's mount Google Drive and install the required packages:

```python
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install pandas torch transformers scikit-learn imbalanced-learn joblib
```

## Extract the Project

Upload the zip file to your Google Drive, then extract it:

```python
# Unzip the project (adjust the path to where you uploaded the zip file)
!unzip /content/drive/MyDrive/bot_detection_project_fixed.zip -d /content/
```

## Run the Training

Now let's run the training with all the fixes implemented:

```python
# Import the training function
from bot_detection_colab import train_model_colab

# Run the training
history, test_results = train_model_colab()
```

## Visualize Results

Let's visualize the training progress:

```python
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot ROC AUC
plt.figure(figsize=(10, 5))
plt.plot(history['roc_auc'], label='ROC AUC')
plt.title('ROC AUC Score')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()

# Plot F1 Score
plt.figure(figsize=(10, 5))
plt.plot(history['f1_score'], label='F1 Score')
plt.title('F1 Score')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()
```

## Test Results

The final test results:

```python
print("Test Results:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")
```
"""

if __name__ == "__main__":
    print("This module provides a complete implementation for running the bot detection model in Google Colab.")
    print("Please upload this file to Colab and import the train_model_colab function.")
