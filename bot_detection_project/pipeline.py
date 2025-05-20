import argparse
import os
import torch
from src.evaluator import ModelEvaluator # Added import
import numpy as np
import json
from datetime import datetime

from src.data_loader import TwitterDataProcessor, create_data_loaders
from src.models.bert_model import BERTBotDetector
from src.models.lstm_model import LSTMBotDetector # Import for potential swapping
from src.models.logistic_regression import LogisticRegressionBotDetector # Import for potential swapping
from src.trainer import ModelTrainer
from src.config import (
    DEVICE, MODEL_NAME, BATCH_SIZE, 
    NUMERICAL_FEATURES, BOOLEAN_FEATURES, USE_OVERSAMPLING, DROPOUT_RATE,
    USE_CLASS_WEIGHTS, MAX_LEN, MODEL_SAVE_DIR
)
from src.utils import set_seed

def main(model_type: str = "bert"):
    """
    Main pipeline script to load data, preprocess, set up the model, and prepare for training.
    
    Args:
        model_type (str): Type of model to use ("bert", "lstm", "logistic_regression"). Default is "bert".
    """
    set_seed() # Ensure reproducibility from the start
    print(f"Using device: {DEVICE}")
    print(f"Using model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_LEN}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dataset-optimized configuration active")

    # 1. Load and preprocess data
    print("\n=== Loading and preprocessing data ===")
    data_processor = TwitterDataProcessor()
    train_df, val_df, test_df, scaler = data_processor.preprocess_data()
    
    # Apply oversampling if configured
    if USE_OVERSAMPLING:
        train_df = data_processor.apply_oversampling()
    
    # Get class weights if needed
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = data_processor.get_class_weights()
    
    # 2. Create data loaders
    print("\n=== Creating data loaders ===")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df,
        data_processor.tokenizer,
        scaler,
        BATCH_SIZE
    )
    
    # Count numerical and boolean features
    num_numerical = len(NUMERICAL_FEATURES) + 4  # +4 for time features
    num_boolean = len(BOOLEAN_FEATURES)
    
    # 3. Initialize model with optimized architecture
    print("\n=== Initializing model ===")
    if model_type.lower() == "bert":
        # Initialize optimized RoBERTa model
        print(f"Using dataset-optimized RoBERTa model: {MODEL_NAME}")
        print(f"Simplified architecture with hidden_size=128")
        model = BERTBotDetector(
            num_numerical_features=num_numerical,
            num_boolean_features=num_boolean,
            bert_model_name=MODEL_NAME,
            dropout_prob=DROPOUT_RATE, # Pass dropout rate from config
            hidden_size=128 # Optimized for dataset complexity
        )
        # Resize model embeddings if we added new tokens
        if data_processor.tokenizer.vocab_size < len(data_processor.tokenizer):
            model.bert.resize_token_embeddings(len(data_processor.tokenizer))
            print(f"Resized token embeddings to {len(data_processor.tokenizer)}")
        optimizer_name = "AdamW"
    elif model_type.lower() == "lstm":
        # Note: LSTM requires input_ids suitable for its embedding layer.
        model = LSTMBotDetector(
            vocab_size=len(data_processor.tokenizer),
            embedding_dim=300,
            hidden_dim=256,
            num_numerical_features=num_numerical,
            num_boolean_features=num_boolean,
            dropout_prob=DROPOUT_RATE
        )
        optimizer_name = "Adam"
    elif model_type.lower() == "logistic_regression":
        # Initialize Logistic Regression with its own embedding layer. embedding_dim_lr is set to 100.
        embedding_dim_lr = 100  # Embedding dimension for Logistic Regression
        vocab_size = len(data_processor.tokenizer)
        model = LogisticRegressionBotDetector(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim_lr,
            num_numerical_features=num_numerical,
            num_boolean_features=num_boolean
        )
        optimizer_name = "Adam"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to device
    model = model.to(DEVICE)
    
    # 4. Create experiment tracking directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME.replace('/', '_')}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    config_dict = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "device": str(DEVICE),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "numerical_features": num_numerical,
        "boolean_features": num_boolean,
        "use_oversampling": USE_OVERSAMPLING,
        "use_class_weights": USE_CLASS_WEIGHTS
    }
    
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Experiment configuration saved to {experiment_dir}")
    
    # 5. Initialize trainer
    print("\n=== Initializing trainer ===")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=optimizer_name,
        class_weights=class_weights
    )
    
    # Return trainer for interactive use or further processing
    return trainer, model, train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot Detection Pipeline")
    parser.add_argument("--model", type=str, default="bert", choices=["bert", "lstm", "logistic_regression"],
                        help="Type of model to use")
    args = parser.parse_args()
    
    # Get trainer and other components
    trainer, model, train_loader, val_loader, test_loader = main(args.model)
    
    # Train the model
    print("\n=== Starting training ===")
    metrics = trainer.train()
    
    # Evaluate on test set using ModelEvaluator
    print("\n=== Evaluating on test set ===")
    best_model_path = trainer.early_stopping.path

    # Evaluate the best model (from early stopping) on the test set using ModelEvaluator for comprehensive metrics.
    if os.path.exists(best_model_path):
        print(f"Loading best model for test evaluation from {best_model_path}")
        # Ensure the model is on the correct device before loading state_dict
        model.to(DEVICE) 
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        evaluator = ModelEvaluator(model=model, test_loader=test_loader, model_path=best_model_path)
        test_metrics, test_report, test_cm = evaluator.evaluate()
    else:
        # Fallback: If best_model.pth is not found, evaluate the model state at the end of training.
        print(f"Warning: Best model path {best_model_path} not found. Evaluating with the model state at the end of training.")
        # MODEL_SAVE_DIR needs to be accessible here. It's imported from config.
        last_model_path = os.path.join(MODEL_SAVE_DIR, 'last_model_state.pth')
        torch.save(model.state_dict(), last_model_path)
        # Ensure the model is on the correct device
        model.to(DEVICE)
        evaluator = ModelEvaluator(model=model, test_loader=test_loader, model_path=last_model_path)
        test_metrics, test_report, test_cm = evaluator.evaluate()

    # The old evaluation call is now replaced by the ModelEvaluator logic above.
    # print("\n=== Evaluating on test set ===")
    # test_loss, test_acc, test_f1, test_precision, test_recall = trainer.evaluate(test_loader)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    # print(f"Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
