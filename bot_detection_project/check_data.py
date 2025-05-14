#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os

# --- Configuration values (copied from config.py for standalone check) ---
_current_dir = os.path.dirname(__file__)
# Corrected: Project root is the directory where this script is located
_project_root = _current_dir 
DATA_PATH = os.path.join(_project_root, "data", "labeled_twitter_data.csv")
TARGET_FEATURE = "Bot Label"
RANDOM_SEED = 123
TEST_SPLIT_SIZE = 0.15
VALID_SPLIT_SIZE = 0.15
# ----------------------------------------------------------------------

def load_data(data_path):
    # Add a check for empty path
    if not data_path:
        raise ValueError("Data path cannot be empty.")
    # Use absolute path for robustness
    abs_data_path = os.path.abspath(data_path)
    print(f"Attempting to load data from absolute path: {abs_data_path}")
    if not os.path.exists(abs_data_path):
        raise FileNotFoundError(f"Data file not found at {abs_data_path}")
    return pd.read_csv(abs_data_path)

def get_class_weights(train_df, target_feature):
    class_counts = train_df[target_feature].value_counts().to_dict()
    total = sum(class_counts.values())
    
    # Handle potential division by zero if a class is missing (shouldn't happen with stratification)
    weights = {cls: total / count if count > 0 else 0 for cls, count in class_counts.items()}
    
    # Normalize weights
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {cls: weight / weight_sum * len(weights) for cls, weight in weights.items()}
    else:
        # If weight_sum is 0 (e.g., empty dataframe), return equal weights
        weights = {cls: 1.0 for cls in class_counts.keys()}
        
    # Ensure weights are in the correct order (0, 1, ...)
    # Handle cases where one class might be missing in a small split (though unlikely with stratification)
    num_classes = train_df[target_feature].nunique()
    weight_list = [0.0] * num_classes
    for i in sorted(weights.keys()):
        if i < num_classes:
             weight_list[i] = weights[i]
        else:
             print(f"Warning: Class index {i} out of bounds for {num_classes} classes.")
             
    weight_tensor = torch.FloatTensor(weight_list)
    return weight_tensor, class_counts

def main():
    # Ensure the script calculates paths relative to its own location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_dir, "data", "labeled_twitter_data.csv")
    print(f"Loading data from: {data_file_path}")
    
    df = load_data(data_file_path)
    print(f"Total samples loaded: {len(df)}")
    print(f"Target distribution in full dataset:\n{df[TARGET_FEATURE].value_counts(normalize=True)}")

    # Split data (mirroring data_loader.py logic)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_SEED,
        stratify=df[TARGET_FEATURE]
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VALID_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE), # Adjust validation split size
        random_state=RANDOM_SEED,
        stratify=train_val_df[TARGET_FEATURE]
    )
    
    print(f"\nTrain set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Calculate class weights based on the training set
    print("\nCalculating class weights based on the training set...")
    weights, counts = get_class_weights(train_df, TARGET_FEATURE)
    
    print(f"Class counts in training set: {counts}")
    print(f"Calculated class weights: {weights}")

if __name__ == "__main__":
    main()


