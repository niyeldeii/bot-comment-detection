import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Import configuration
from .config import (
    DATA_PATH, MODEL_NAME, MAX_LEN, BATCH_SIZE, RANDOM_SEED,
    TEST_SPLIT_SIZE, VALID_SPLIT_SIZE, DEVICE,
    TEXT_FEATURE, NUMERICAL_FEATURES, BOOLEAN_FEATURES, TARGET_FEATURE
)

class TwitterDataProcessor:
    """
    Class for preprocessing Twitter data for bot detection
    """
    def __init__(self, data_path: str = DATA_PATH):
        """
        Initialize the data processor
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        print(f"Initializing tokenizer for {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Add special tokens for social media content that work across platforms
        special_tokens = {'additional_special_tokens': ['<url>', '<user>', '<hashtag>']}
        self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added cross-platform special tokens: {special_tokens['additional_special_tokens']}")
        self.scaler = StandardScaler()
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        # Define numerical columns including time features
        self.numerical_cols_with_time = NUMERICAL_FEATURES + ["hour", "day_of_week", "month", "year"]
        # Define only the base numerical features for log transform
        self.numerical_cols_to_log_transform = NUMERICAL_FEATURES
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from CSV file
        
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"Data file not found at {self.data_path}. Please ensure the path is correct in config.py and the file exists.")
        self.df = pd.read_csv(self.data_path)
        # Ensure numerical columns are numeric, coercing errors
        for col in NUMERICAL_FEATURES:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        return self.df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text) # Ensure text is string
            
        # Simplified preprocessing focused on key bot indicators
        text = re.sub(r"https?://\S+|www\.\S+", "<url>", text)
        
        text = re.sub(r"@\w+", "<user>", text)
        
        text = re.sub(r"#(\w+)", r"<hashtag> \1", text)
        
        # Detect repetitive patterns (common in bot tweets)
        text = re.sub(r"(\b\w+\b)\s+\1+", r"\1 <repeated>", text)  # Mark repeated words
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def extract_time_features(self, created_at: str) -> Dict[str, int]:
        """
        Extract time-based features from the "Created At" column
        
        Args:
            created_at: Timestamp string
            
        Returns:
            Dictionary of time features
        """
        # Direct parsing without try/except for cleaner, faster code
        # Dataset has consistent date format
        dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        return {
            "hour": dt.hour,
            "day_of_week": dt.weekday(),
            "month": dt.month,
            "year": dt.year - 2020  # Normalize years (2020-2023)
        }
    
    def extract_time_features_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from the "Created At" column for a DataFrame
        
        Args:
            df: DataFrame with "Created At" column
            
        Returns:
            DataFrame with time features
        """
        time_features = df["Created At"].apply(self.extract_time_features)
        time_df = pd.DataFrame(time_features.tolist())
        df = pd.concat([df, time_df], axis=1)
        return df
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Preprocess the data, split into train/val/test, and fit the scaler.
        
        Returns:
            Tuple of (train_df, val_df, test_df, fitted_scaler)
        """
        print("Loading and preprocessing data...")
        self.load_data()
        
        # Extract time features
        print("Extracting time features...")
        self.df = self.extract_time_features_for_df(self.df)
        
        # Clean text
        self.df['cleaned_tweet'] = self.df[TEXT_FEATURE].apply(self.preprocess_text)
        
        # Print class distribution before preprocessing
        print("Class distribution before preprocessing:")
        print(self.df[TARGET_FEATURE].value_counts(normalize=True))
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            print("Filling missing values...")
            # Fill missing numerical values with median
            for col in self.numerical_cols_with_time:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
            # Fill missing text with empty string
            self.df[TEXT_FEATURE] = self.df[TEXT_FEATURE].fillna('')
            # Fill missing boolean with 0
            for col in BOOLEAN_FEATURES:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(0)
        
        # Clip extreme values at 99.9 percentile to prevent outliers from affecting scaling
        for col in self.numerical_cols_to_log_transform:
            # Calculate 99.9 percentile
            p999 = np.percentile(self.df[col], 99.9)
            # Clip values
            self.df[col] = self.df[col].clip(upper=p999)
            # Apply log1p transformation
            self.df[col] = np.log1p(self.df[col])
        
        # Also clip other numerical features that aren't log-transformed
        for col in self.numerical_cols_with_time:
            if col not in self.numerical_cols_to_log_transform:
                # Calculate 99.9 and 0.1 percentiles
                p999 = np.percentile(self.df[col], 99.9)
                p001 = np.percentile(self.df[col], 0.1)
                # Clip values
                self.df[col] = self.df[col].clip(lower=p001, upper=p999)
        
        # Split data into train, validation, and test sets
        print("Splitting data...")
        train_df, temp_df = train_test_split(
            self.df, test_size=0.3, random_state=RANDOM_SEED, stratify=self.df[TARGET_FEATURE]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df[TARGET_FEATURE]
        )
        
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        
        # Print split sizes
        print(f"Train set size: {len(self.train_df)} ({len(self.train_df[self.train_df[TARGET_FEATURE] == 1])} positive)")
        print(f"Validation set size: {len(self.val_df)} ({len(self.val_df[self.val_df[TARGET_FEATURE] == 1])} positive)")
        print(f"Test set size: {len(self.test_df)} ({len(self.test_df[self.test_df[TARGET_FEATURE] == 1])} positive)")
        
        # Fit scaler on training data
        print(f"Fitting scaler on training data columns: {self.numerical_cols_with_time}")
        # Check if all columns exist in the training data
        missing_cols = [col for col in self.numerical_cols_with_time if col not in self.train_df.columns]
        if missing_cols:
            raise ValueError(f"Scaler fitting error: Missing columns in training data: {missing_cols}")
        self.scaler.fit(self.train_df[self.numerical_cols_with_time])
        
        # Save feature names in the scaler for later use
        self.scaler.feature_names_in_ = np.array(self.numerical_cols_with_time)
        
        print(f"Preprocessing complete. Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        return self.train_df, self.val_df, self.test_df, self.scaler
    
    def apply_oversampling(self) -> pd.DataFrame:
        """
        Apply oversampling to the training data to balance classes
        
        Returns:
            Oversampled training DataFrame
        """
        if self.train_df is None:
            raise ValueError("Training data not available. Call preprocess_data first.")
            
        print("Applying RandomOverSampler oversampling to balance classes...")
        
        # Print class distribution before oversampling
        print("Class distribution before oversampling:")
        print(self.train_df[TARGET_FEATURE].value_counts(normalize=True))
        
        # Get features (exclude text features which SMOTE can't handle)
        X = self.train_df.drop([TARGET_FEATURE, TEXT_FEATURE, 'cleaned_tweet'], axis=1)
        y = self.train_df[TARGET_FEATURE]
        
        # Store text features separately
        text_features = self.train_df[[TEXT_FEATURE, 'cleaned_tweet']]
        
        # Use RandomOverSampler with a more conservative sampling strategy
        ros = RandomOverSampler(random_state=RANDOM_SEED, sampling_strategy=0.8)  # 80% of majority class
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        # Create new DataFrame with oversampled data (without text features)
        oversampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        oversampled_df[TARGET_FEATURE] = y_resampled
        
        # Get indices of original samples
        original_indices = np.arange(len(self.train_df))
        
        # Get indices of synthetic samples
        synthetic_indices = np.arange(len(self.train_df), len(oversampled_df))
        
        # For original samples, copy text features from original DataFrame
        original_oversampled_df = oversampled_df.iloc[original_indices].copy()
        original_oversampled_df[TEXT_FEATURE] = text_features[TEXT_FEATURE].values
        original_oversampled_df['cleaned_tweet'] = text_features['cleaned_tweet'].values
        
        # For synthetic samples, find nearest neighbors in original data and use their text
        synthetic_oversampled_df = oversampled_df.iloc[synthetic_indices].copy()
        
        # Find minority class samples in original data
        minority_indices = self.train_df[self.train_df[TARGET_FEATURE] == 1].index
        minority_text = text_features.iloc[minority_indices]
        
        # Randomly assign text from minority class to synthetic samples
        random_minority_indices = np.random.choice(
            minority_indices, 
            size=len(synthetic_indices), 
            replace=True
        )
        
        synthetic_oversampled_df[TEXT_FEATURE] = text_features.iloc[random_minority_indices][TEXT_FEATURE].values
        synthetic_oversampled_df['cleaned_tweet'] = text_features.iloc[random_minority_indices]['cleaned_tweet'].values
        
        # Combine original and synthetic samples
        oversampled_df = pd.concat([original_oversampled_df, synthetic_oversampled_df], axis=0)
        
        # Print class distribution after oversampling
        print("Class distribution after oversampling:")
        print(oversampled_df[TARGET_FEATURE].value_counts(normalize=True))
        
        # Update training data
        self.train_df = oversampled_df
        
        return oversampled_df
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights based on class distribution
        
        Returns:
            Tensor of class weights
        """
        if self.train_df is None:
            raise ValueError("Training data not available. Call preprocess_data first.")
            
        print("Calculating class weights...")
        
        # Get class counts
        class_counts = self.train_df[TARGET_FEATURE].value_counts()
        print(f"Class distribution: {class_counts}")
        
        # Calculate weights using 'balanced' approach
        n_samples = len(self.train_df)
        n_classes = len(class_counts)
        
        weights = n_samples / (n_classes * class_counts)
        
        # Normalize weights to prevent extreme values
        weights = weights / weights.sum() * n_classes
        
        # Convert to tensor
        class_weights = torch.tensor([
            weights.get(0, 1.0),  # Weight for class 0
            weights.get(1, 1.0)   # Weight for class 1
        ], dtype=torch.float)
        
        print(f"Calculated class weights: {class_weights}")
        
        return class_weights


class TwitterDataset(Dataset):
    """
    PyTorch Dataset for Twitter data
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        scaler: StandardScaler, # Accept the fitted scaler
        max_len: int = MAX_LEN,
        is_test: bool = False
    ):
        """
        Initialize the dataset
        
        Args:
            dataframe: Preprocessed DataFrame for this split (train/val/test)
            tokenizer: RoBERTa tokenizer
            scaler: Pre-fitted StandardScaler
            max_len: Maximum sequence length
            is_test: Whether this is a test dataset (no targets)
        """
        self.dataframe = dataframe.reset_index(drop=True) # Ensure index is continuous
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.max_len = max_len
        self.is_test = is_test
        
        # Define numerical columns including time features
        self.numerical_cols_to_scale = NUMERICAL_FEATURES + ["hour", "day_of_week", "month", "year"]
        
        # Ensure columns exist before accessing them
        missing_num_cols = [col for col in self.numerical_cols_to_scale if col not in self.dataframe.columns]
        if missing_num_cols:
            raise ValueError(f"Dataset Init Error: Missing numerical columns in dataframe: {missing_num_cols}")
            
        missing_bool_cols = [col for col in BOOLEAN_FEATURES if col not in self.dataframe.columns]
        if missing_bool_cols:
            raise ValueError(f"Dataset Init Error: Missing boolean columns in dataframe: {missing_bool_cols}")
            
        if not self.is_test and TARGET_FEATURE not in self.dataframe.columns:
            raise ValueError(f"Dataset Init Error: Target column ", TARGET_FEATURE, " not found in dataframe.")
            
        # Store data directly instead of scaling here
        self.numerical_data = self.dataframe[self.numerical_cols_to_scale].values
        self.boolean_data = self.dataframe[BOOLEAN_FEATURES].astype(float).values # Ensure float for tensor
        self.texts = self.dataframe["cleaned_tweet"].values
        if not self.is_test:
            self.targets = self.dataframe[TARGET_FEATURE].values
        
    def __len__(self) -> int:
        """
        Get the length of the dataset
        
        Returns:
            Number of samples
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary of tensors
        """
        text = str(self.texts[idx]) # Ensure text is string
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,  # Include token type IDs
            return_tensors="pt"
        )
        
        # Apply scaling here
        # Reshape numerical data for scaler (expects 2D array)
        numerical_features_unscaled = self.numerical_data[idx].reshape(1, -1)
        
        # Handle feature names correctly to avoid warnings
        if hasattr(self.scaler, 'feature_names_in_'):
            # Create a DataFrame with proper column names
            numerical_df = pd.DataFrame(
                numerical_features_unscaled, 
                columns=self.scaler.feature_names_in_
            )
            numerical_features_scaled = self.scaler.transform(numerical_df)
        else:
            # Fallback if feature names are not available
            numerical_features_scaled = self.scaler.transform(numerical_features_unscaled)
        
        numerical_features = torch.FloatTensor(numerical_features_scaled.flatten()) # Flatten back to 1D tensor
        
        boolean_features = torch.FloatTensor(self.boolean_data[idx])
        
        # Combine all features
        sample = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "numerical_features": numerical_features,
            "boolean_features": boolean_features
        }
        
        if not self.is_test:
            sample["targets"] = torch.tensor(self.targets[idx], dtype=torch.long)
            
        return sample


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    scaler: StandardScaler, # Pass the fitted scaler
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: RoBERTa tokenizer
        scaler: Fitted StandardScaler
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TwitterDataset(train_df, tokenizer, scaler, max_len=MAX_LEN)
    val_dataset = TwitterDataset(val_df, tokenizer, scaler, max_len=MAX_LEN)
    # Test dataset might not have targets, handle accordingly if needed
    test_dataset = TwitterDataset(test_df, tokenizer, scaler, max_len=MAX_LEN, is_test=True) # Set is_test=True for test loader
    
    # Consider using persistent_workers=True if num_workers > 0 for potential speedup
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader
