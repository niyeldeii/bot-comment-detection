# Configuration settings for the bot detection project

import torch
import os

# Get the directory where this config file is located
_current_dir = os.path.dirname(__file__)
# Get the project root directory (one level up from src)
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))

# Data paths - Using absolute paths derived from the project structure for robustness
DATA_PATH = os.path.join(_project_root, "data", "labeled_twitter_data.csv")
MODEL_SAVE_DIR = os.path.join(_project_root, "models")
RESULTS_DIR = os.path.join(_project_root, "results")

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model parameters
MODEL_NAME = "roberta-large" # Larger RoBERTa model for optimal cross-platform performance
MAX_LEN = 128 # Optimized sequence length for the dataset's short tweets
BATCH_SIZE = 8 
EPOCHS = 20 
DROPOUT_RATE = 0.3 # Moderate dropout for balanced data

# --- FIX: Adjusted Learning Rates --- 
LEARNING_RATE = 5e-4  # Moderate learning rate for classifier head
BERT_LR = 2e-5  # Careful fine-tuning rate for large BERT model
# ----------------------------------

# --- Scheduler Configuration ---
SCHEDULER_TYPE = 'linear'  # Linear scheduler for consistent learning
WARMUP_STEPS = 0  # No warmup - immediate full learning rate
SCHEDULER_PATIENCE = 2  # Reduced patience

# Early Stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Increased to give model more time with the new settings

# Features
TEXT_FEATURE = "Tweet"
NUMERICAL_FEATURES = ["Retweet Count", "Follower Count"] # Removed Mention Count as it's less predictive
BOOLEAN_FEATURES = ["Verified"]
TARGET_FEATURE = "Bot Label"

# Other settings
RANDOM_SEED = 123 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SPLIT_SIZE = 0.15
VALID_SPLIT_SIZE = 0.15 # Of the remaining data after test split

# Class weighting and balancing
USE_CLASS_WEIGHTS = False  # Disable class weights since data is balanced
CLASS_WEIGHTS = None

# Oversampling (alternative to class weights)
USE_OVERSAMPLING = False  # Disable oversampling since data is balanced

# Weight decay for optimizer (L2 regularization) - minimal weight decay
WEIGHT_DECAY = 0.0001 

# --- FIX: Adjusted Gradient Clipping --- 
GRAD_CLIP = 1.0 # Standard gradient clipping value
# ---------------------------------------

# --- Training Dynamics Configuration ---
GRAD_ACCUMULATION_STEPS = 1  # Update after every batch
# ---------------------------------------

# Freeze BERT layers - unfreeze all layers for maximum learning capacity
FREEZE_BERT_LAYERS = 0 
