import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import os
import time
from typing import Dict, Tuple, Optional, List

from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE, BERT_LR, MODEL_SAVE_DIR, CLASS_WEIGHTS, 
    WARMUP_STEPS, WEIGHT_DECAY, GRAD_CLIP, GRAD_ACCUMULATION_STEPS, FREEZE_BERT_LAYERS, USE_OVERSAMPLING,
    USE_CLASS_WEIGHTS, SCHEDULER_TYPE, SCHEDULER_PATIENCE, EARLY_STOPPING_PATIENCE # Import parameters
)
from src.utils import set_seed

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=4, reduction='mean', weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.001, path: str = 'checkpoint.pt', trace_func=print):
        """
        Initialize early stopping with improved parameters
        
        Args:
            patience: How many epochs to wait after last improvement
            verbose: If True, prints a message for each validation loss improvement
            delta: Minimum change in monitored quantity to qualify as improvement
            path: Path to save the checkpoint
            trace_func: Function to print messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ModelTrainer:
    """
    Class to handle the training and validation process with improved training dynamics
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_name: str = 'AdamW',
        use_scheduler: bool = True,
        use_early_stopping: bool = True,
        patience: int = 7,
        class_weights: Optional[torch.Tensor] = None
    ):
        set_seed()
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_scheduler = use_scheduler
        self.use_early_stopping = use_early_stopping
        
        # --- Freezing BERT layers logic (controlled by config) --- 
        if hasattr(self.model, 'bert'):
            print("Freezing RoBERTa embeddings layer")
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
                
            
            num_layers = len(self.model.bert.encoder.layer)
            layers_to_freeze = int(num_layers * 5/6) 
            
            print(f"RoBERTa has {num_layers} layers. Freezing first {layers_to_freeze} layers.")
            for i, layer in enumerate(self.model.bert.encoder.layer):
                if i < layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
                    print(f"Layer {i} is trainable")
            
            if hasattr(self.model.bert, 'pooler'):
                for param in self.model.bert.pooler.parameters():
                    param.requires_grad = True
                print("RoBERTa pooler is trainable")
        # ----------------------------------------
        
        # Setup loss function with optional class weights
        if USE_CLASS_WEIGHTS and class_weights is not None:
            print(f"Using class weights: {class_weights}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
            print("Using CrossEntropyLoss with class weights and label smoothing=0.1")
        else:
            print("Using standard CrossEntropyLoss for balanced data.")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            print("Added label smoothing=0.1 for better generalization with large model")
            
        # --- Setup Optimizer with Differential Learning Rates --- 
        if hasattr(self.model, 'bert'):
            print(f"Setting up optimizer with differential learning rates: BERT LR={BERT_LR}, Head LR={LEARNING_RATE}")
            # Define parameter groups
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.bert.named_parameters() if p.requires_grad],
                    'lr': BERT_LR,
                    'weight_decay': WEIGHT_DECAY
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if not n.startswith('bert.') and p.requires_grad],
                    'lr': LEARNING_RATE,
                    'weight_decay': WEIGHT_DECAY
                }
            ]
            # Filter out groups with no parameters requiring gradients
            optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if len(group['params']) > 0]
            
        else: # For non-BERT models (LSTM, LR), use the main LEARNING_RATE
            print(f"Setting up optimizer with standard learning rate: {LEARNING_RATE}")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
            ]
        
        # Initialize optimizer
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        elif optimizer_name.lower() == 'adam':
             self.optimizer = optim.Adam(optimizer_grouped_parameters)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        # --------------------------------------------------------
            
        # Setup scheduler with warmup steps
        if self.use_scheduler:
            if SCHEDULER_TYPE == 'linear':
                num_update_steps_per_epoch = len(self.train_loader)
                total_steps = num_update_steps_per_epoch * EPOCHS
                print(f"Setting up linear scheduler with {WARMUP_STEPS} warmup steps and {total_steps} total steps.")
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=WARMUP_STEPS,
                    num_training_steps=total_steps
                )
            elif SCHEDULER_TYPE == 'plateau':
                print("Setting up ReduceLROnPlateau scheduler.")
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)
            else:
                raise ValueError(f"Unsupported scheduler type: {SCHEDULER_TYPE}")
        else:
            self.scheduler = None
            
        # Setup early stopping
        if self.use_early_stopping:
            model_save_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
            self.early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=model_save_path)
        else:
            self.early_stopping = None
            
    def train_epoch(self) -> float:
        """
        Train for one epoch with gradient accumulation
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        # Reduce gradient accumulation for simpler dataset
        accumulation_steps = 1  # No need for gradient accumulation with simplified model
        
        self.optimizer.zero_grad() # Zero gradients at the start of the epoch
        
        # Track statistics for monitoring
        correct_predictions = 0
        total_predictions = 0
        all_targets = []
        all_predictions = []
        
        # Track class-wise accuracy
        class_correct = {0: 0, 1: 0}
        class_total = {0: 0, 1: 0}
        
        for batch_num, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            boolean_features = batch['boolean_features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                numerical_features=numerical_features,
                boolean_features=boolean_features
            )
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Track accuracy statistics
            _, predictions = torch.max(outputs, dim=1)
            
            # Track class-wise accuracy
            for c in [0, 1]:
                class_mask = (targets == c)
                class_total[c] += class_mask.sum().item()
                class_correct[c] += ((predictions == c) & class_mask).sum().item()
            
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
            # Unscaled loss for reporting
            total_loss += loss.item() * targets.size(0) # Loss per sample
            
            # Step optimizer and scheduler every accumulation_steps batches
            if (batch_num + 1) % accumulation_steps == 0 or (batch_num + 1) == len(self.train_loader):
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRAD_CLIP)
                
                # Log gradient norm occasionally
                if (batch_num + 1) % 100 == 0:
                    print(f"  Batch {batch_num + 1}/{len(self.train_loader)}, Gradient norm: {grad_norm:.4f}")
                
                # Optimizer step
                self.optimizer.step()
                
                # Step the linear scheduler (if used)
                if self.scheduler and SCHEDULER_TYPE == 'linear':
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
            
            # Print progress
            if (batch_num + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                # Calculate training accuracy so far
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                # Calculate class-wise accuracy
                class0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
                class1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0
                print(f"  Batch {batch_num + 1}/{len(self.train_loader)}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.4f}, Class0: {class0_acc:.4f}, Class1: {class1_acc:.4f}, Time: {elapsed_time:.2f}s")
        
        # Calculate final training metrics
        train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        class0_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        class1_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        print(f"  Class-wise accuracy - Non-bots: {class0_acc:.4f}, Bots: {class1_acc:.4f}")
        
        # Calculate class distribution in predictions
        if len(all_predictions) > 0:
            unique, counts = np.unique(all_predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))
            print(f"  Prediction distribution: {pred_distribution}")
            if len(pred_distribution) < 2 or min(counts)/max(counts) < 0.1:
                print("  WARNING: Model is predicting mostly one class! Check feature quality and model initialization.")
        
        # Calculate average loss per sample
        avg_train_loss = total_loss / len(self.train_loader.dataset) # Loss per sample
        print(f"  Training accuracy: {train_accuracy:.4f}")
        
        return avg_train_loss
        
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on validation data
        
        Returns:
            Tuple of (validation loss, metrics dictionary)
        """
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                token_type_ids = batch['token_type_ids'].to(DEVICE)
                numerical_features = batch['numerical_features'].to(DEVICE)
                boolean_features = batch['boolean_features'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    numerical_features=numerical_features,
                    boolean_features=boolean_features
                )
                
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0) # Multiply by batch size for total loss
                
                _, predictions = torch.max(outputs, dim=1)
                
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(self.val_loader.dataset)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')
        
        try:
            roc_auc = roc_auc_score(all_targets, all_predictions)
        except ValueError:
            roc_auc = 0.5  # Default value when ROC AUC can't be calculated
            print("Warning: ROC AUC could not be calculated (possibly only one class predicted)")
        
        unique, counts = np.unique(all_predictions, return_counts=True)
        pred_distribution = dict(zip(unique, counts))
        print(f"  Validation prediction distribution: {pred_distribution}")
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }
        
        return avg_val_loss, metrics
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model with logging and checkpointing
        
        Returns:
            Dictionary of training history
        """
        history = {
            'train_loss': [], 'val_loss': [], 
            'precision': [], 'recall': [], 'f1_score': [], 'roc_auc': [],
            'learning_rates': [], 'grad_norms': [], 'train_accuracy': [], 'val_accuracy': []
        }
        
        print("Starting training...")
        
        print(f"Model is on device: {next(self.model.parameters()).device}")
        
        # Analyze feature importance before training
        self._analyze_feature_importance()
        
        for batch in self.train_loader:
            print(f"Data is on device: {batch['input_ids'].device}")
            break
        
        frozen_layers = []
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            # Initially all trainable layers are stored
            for i, layer in enumerate(self.model.bert.encoder.layer):
                if not any(param.requires_grad for param in layer.parameters()):
                    frozen_layers.append(i)
            print(f"Initially frozen layers: {frozen_layers}")
        
        learning_capacity_warning = False
        stagnation_counter = 0
        min_improvement_threshold = 0.001  # Minimum expected improvement
        prev_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            print(f"\n{'='*20} Epoch {epoch + 1}/{EPOCHS} {'='*20}")
            start_epoch_time = time.time()
            
            avg_train_loss = self.train_epoch()
            history['train_loss'].append(avg_train_loss)
            
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            history['learning_rates'].append(lrs)
            
            val_loss, val_metrics = self.evaluate()
            history['val_loss'].append(val_loss)
            history['precision'].append(val_metrics['precision'])
            history['recall'].append(val_metrics['recall'])
            history['f1_score'].append(val_metrics['f1_score'])
            history['roc_auc'].append(val_metrics['roc_auc'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            if self.scheduler and SCHEDULER_TYPE == 'plateau':
                self.scheduler.step(val_loss)
                lrs = [group['lr'] for group in self.optimizer.param_groups]
                print(f"  Learning Rates: {', '.join([f'{lr:.2e}' for lr in lrs])}")
            
            # Gradually unfreeze more layers as training progresses
            if hasattr(self.model, 'bert') and epoch > 0 and epoch % 3 == 0 and frozen_layers:
                # Unfreeze one more layer every 3 epochs
                layer_to_unfreeze = frozen_layers.pop()
                print(f"Unfreezing layer {layer_to_unfreeze} at epoch {epoch+1}")
                for param in self.model.bert.encoder.layer[layer_to_unfreeze].parameters():
                    param.requires_grad = True
            
            if epoch > 0:
                improvement = prev_val_loss - val_loss
                if improvement < min_improvement_threshold:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                if stagnation_counter >= 3 and not learning_capacity_warning:
                    learning_capacity_warning = True
                    print("\n⚠️ WARNING: Model may be struggling to learn. Consider increasing model capacity.")
                    print("Potential solutions:")
                    print("1. Increase hidden_size from 128 to 256")
                    print("2. Unfreeze more RoBERTa layers")
                    print("3. Add back the second projection layer")
                    print("4. Increase learning rate slightly")
                    
                    # Automatically unfreeze one more layer if available
                    if frozen_layers:
                        layer_to_unfreeze = frozen_layers.pop()
                        print(f"\nAutomatically unfreezing layer {layer_to_unfreeze} to increase capacity")
                        for param in self.model.bert.encoder.layer[layer_to_unfreeze].parameters():
                            param.requires_grad = True
            
            prev_val_loss = val_loss
            
            epoch_duration = time.time() - start_epoch_time
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.6f} (per sample)")
            print(f"  Val Loss: {val_loss:.6f} (per sample)")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1 Score: {val_metrics['f1_score']:.4f}")
            print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Learning Rates: {', '.join([f'{lr:.2e}' for lr in lrs])}")
            print(f"  Duration: {epoch_duration:.2f}s")
            
            # Analyze feature importance every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._analyze_feature_importance()
            
            if epoch >= 2:
                if history['train_loss'][-1] > 0.99 * history['train_loss'][0]:
                    print("\nWARNING: Model may not be learning - training loss not decreasing significantly")
                    print("Consider adjusting learning rates or checking for data issues")
                    
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, grad_mean={param.grad.mean().item() if param.grad is not None else 0:.6f}")
                
                if abs(history['train_loss'][-1] - history['val_loss'][-1]) > 0.3:
                    print("\nWARNING: Large gap between training and validation loss - possible overfitting")
                    print("Consider increasing dropout or weight decay")
            
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'metrics': val_metrics
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
            
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                    
        print("\nTraining finished.")
        if self.use_early_stopping and self.early_stopping.best_score is not None:
            best_model_path = self.early_stopping.path
            if os.path.exists(best_model_path):
                print(f"Loading best model state from: {best_model_path}")
                try:
                    self.model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
                except Exception as e:
                    print(f"Error loading best model state_dict from {best_model_path}: {e}")
            else:
                 print(f"Warning: Early stopping was enabled, but best model file not found at {best_model_path}")
        elif self.use_early_stopping:
            print("Warning: Early stopping enabled but no improvement detected, best model not saved.")
            
        return history

    def _analyze_feature_importance(self):
        """
        Analyze feature importance to diagnose learning issues
        """
        print("\nAnalyzing feature importance...")
        self.model.eval()
        
        # Get a batch of data
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            boolean_features = batch['boolean_features'].to(DEVICE)
            targets = batch['targets'].to(DEVICE)
            break
        
        # Analyze BERT features
        with torch.no_grad():
            # Get BERT output
            bert_output = self.model.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_token = bert_output.last_hidden_state[:, 0]
            
            # Get feature importance from weights
            if hasattr(self.model, 'bert_linear'):
                bert_weights = self.model.bert_linear.weight.abs().mean(dim=0)
                top_bert_indices = torch.argsort(bert_weights, descending=True)[:10]
                print("Top 10 BERT feature indices:", top_bert_indices.cpu().numpy())
                print("Top 10 BERT feature weights:", bert_weights[top_bert_indices].cpu().numpy())
            
            # Analyze numerical features
            if hasattr(self.model, 'numerical_linear'):
                num_weights = self.model.numerical_linear.weight.abs().mean(dim=0)
                top_num_indices = torch.argsort(num_weights, descending=True)
                print("\nNumerical feature importance:")
                for i, idx in enumerate(top_num_indices.cpu().numpy()):
                    print(f"  Feature {idx}: {num_weights[idx].item():.4f}")
            
            # Analyze boolean features
            if hasattr(self.model, 'boolean_linear'):
                bool_weights = self.model.boolean_linear.weight.abs().mean(dim=0)
                top_bool_indices = torch.argsort(bool_weights, descending=True)
                print("\nBoolean feature importance:")
                for i, idx in enumerate(top_bool_indices.cpu().numpy()):
                    print(f"  Feature {idx}: {bool_weights[idx].item():.4f}")
        
        # Check for dead neurons
        print("\nChecking for dead neurons...")
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Check for dead neurons (rows with all zeros)
                dead_neurons = (param.abs().sum(dim=1) < 1e-6).sum().item()
                if dead_neurons > 0:
                    print(f"  {name}: {dead_neurons} dead neurons out of {param.size(0)}")
        
        # Check for activation statistics
        print("\nChecking activation statistics...")
        self.model.train()  # Set back to train mode
