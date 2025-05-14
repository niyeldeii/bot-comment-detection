import torch
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import os
import joblib # For loading the scaler
from typing import List, Dict, Tuple

from src.config import (
    DEVICE, MODEL_NAME, MAX_LEN, MODEL_SAVE_DIR, 
    NUMERICAL_FEATURES, BOOLEAN_FEATURES, TEXT_FEATURE
)
from src.models.bert_model import BERTBotDetector # Assuming BERT is the default trained model
from src.data_loader import TwitterDataProcessor, SCALER_PATH # Import SCALER_PATH

class Predictor:
    """
    Class to load a trained model and make predictions on new data
    """
    def __init__(
        self,
        model_path: str = os.path.join(MODEL_SAVE_DIR, "best_model.pth"), # Path to the best saved model
        scaler_path: str = SCALER_PATH, # Path to the saved scaler
        num_numerical_features: int = len(NUMERICAL_FEATURES) + 4, # Base + time features
        num_boolean_features: int = len(BOOLEAN_FEATURES)
    ):
        """
        Initialize the Predictor
        
        Args:
            model_path: Path to the saved model weights (.pth file)
            scaler_path: Path to the saved scaler (.joblib file)
            num_numerical_features: Number of numerical features the model expects
            num_boolean_features: Number of boolean features the model expects
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        # Use a temporary instance for preprocessing functions, doesn't need data path here
        self.data_processor = TwitterDataProcessor(data_path=None) 
        self.num_numerical_features = num_numerical_features
        self.num_boolean_features = num_boolean_features
        
        # Load the scaler
        self.load_scaler()
        
        # Instantiate the model architecture
        # Ensure this matches the architecture of the saved model
        self.model = BERTBotDetector(
            num_numerical_features=self.num_numerical_features,
            num_boolean_features=self.num_boolean_features
            # Dropout rate is now taken from config within the model definition
        )
        
        # Load the trained weights
        self.load_model()
        self.model.to(DEVICE)
        self.model.eval()
        
    def load_scaler(self):
        """Load the fitted StandardScaler."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}. Ensure the model has been trained (which saves the scaler). Run pipeline.py first (even without training) to generate the scaler.")
        self.scaler = joblib.load(self.scaler_path)
        print(f"Scaler loaded successfully from {self.scaler_path}")
        
    def load_model(self):
        """Load the trained model weights."""
        if not os.path.exists(self.model_path):
            # Check if the scaler exists, if not, pipeline likely wasn't run
            if not os.path.exists(self.scaler_path):
                 raise FileNotFoundError(f"Model weights not found at {self.model_path} and scaler not found at {self.scaler_path}. Please run pipeline.py first to preprocess data and save the scaler. Then, train the model to save weights.")
            else:
                 # Scaler exists, but model doesn't - means training likely hasn't happened
                 raise FileNotFoundError(f"Model weights not found at {self.model_path}. Ensure the model is trained and saved (by uncommenting trainer.train() in pipeline.py and running it).")
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        print(f"Model loaded successfully from {self.model_path}")
        
    def preprocess_input(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a list of input dictionaries (representing tweets/comments)
        
        Args:
            data: List of dictionaries, each containing fields like "Tweet", "Retweet Count", etc.
            
        Returns:
            Dictionary of batched tensors ready for the model
        """
        df = pd.DataFrame(data)
        
        # Apply text cleaning
        df["cleaned_tweet"] = df[TEXT_FEATURE].apply(self.data_processor.preprocess_text)
        
        # Extract time features (assuming "Created At" is provided)
        if "Created At" in df.columns:
            time_features = df["Created At"].apply(self.data_processor.extract_time_features)
            time_df = pd.DataFrame(time_features.tolist())
            df = pd.concat([df, time_df], axis=1)
        else:
            # Add default time features if "Created At" is missing
            for feature in ["hour", "day_of_week", "month", "year"]:
                df[feature] = 0
                
        # Ensure all required numerical and boolean features are present
        required_numerical = self.data_processor.numerical_cols_to_scale
        required_boolean = BOOLEAN_FEATURES
        
        for col in required_numerical:
            if col not in df.columns:
                print(f"Warning: Numerical feature ", col, " missing in input data. Using default value 0.")
                df[col] = 0 # Default value if missing
        for col in required_boolean:
            if col not in df.columns:
                print(f"Warning: Boolean feature ", col, " missing in input data. Using default value False.")
                df[col] = False # Default value if missing
                
        # Tokenize text
        texts = df["cleaned_tweet"].tolist()
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Prepare numerical features - Apply the loaded scaler
        numerical_data = df[required_numerical].values.astype(np.float32)
        try:
            numerical_data_scaled = self.scaler.transform(numerical_data)
        except Exception as e:
             print(f"Error applying scaler transform: {e}. Using unscaled data.")
             numerical_data_scaled = numerical_data # Fallback, though ideally should match training
             
        numerical_features_tensor = torch.tensor(numerical_data_scaled, dtype=torch.float)
        
        # Prepare boolean features
        boolean_data = df[required_boolean].astype(int).values
        boolean_features_tensor = torch.tensor(boolean_data, dtype=torch.float)
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "numerical_features": numerical_features_tensor,
            "boolean_features": boolean_features_tensor
        }

    def predict(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a list of new data points
        
        Args:
            data: List of dictionaries representing new tweets/comments
            
        Returns:
            Tuple of (predictions (0 or 1), probabilities of being a bot)
        """
        if not data:
            return np.array([]), np.array([])
            
        processed_input = self.preprocess_input(data)
        
        # Move tensors to device
        input_ids = processed_input["input_ids"].to(DEVICE)
        attention_mask = processed_input["attention_mask"].to(DEVICE)
        numerical_features = processed_input["numerical_features"].to(DEVICE)
        boolean_features = processed_input["boolean_features"].to(DEVICE)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                boolean_features=boolean_features
            )
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        # Return predictions (0 or 1) and probability of class 1 (Bot)
        return predictions.cpu().numpy(), probabilities[:, 1].cpu().numpy()

# Example Usage (can be run as a script or imported)
if __name__ == "__main__":
    # Ensure the model is trained and saved at the expected path first
    # Running pipeline.py (even without training) is needed to create the scaler file.
    model_save_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    scaler_save_path = SCALER_PATH
    
    # Check if scaler exists (means pipeline was run at least once)
    if not os.path.exists(scaler_save_path):
        print(f"Error: Scaler file not found at {scaler_save_path}")
        print("Please run pipeline.py first to preprocess data and save the scaler.")
    else:
        # Check if model exists (means training was likely run)
        if not os.path.exists(model_save_path):
            print(f"Warning: Model file not found at {model_save_path}")
            print("Inference will use initial model weights. Train the model for meaningful predictions.")
            # Create a dummy model file for demonstration if it doesn't exist
            print("Creating a dummy model file for demonstration purposes...")
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            dummy_model = BERTBotDetector(num_numerical_features=len(NUMERICAL_FEATURES) + 4, num_boolean_features=len(BOOLEAN_FEATURES))
            torch.save(dummy_model.state_dict(), model_save_path)
            print(f"Dummy model saved to {model_save_path}")

        # Sample new data (replace with actual data)
        new_data = [
            {
                "Tweet": "This is a great post! Very informative.", 
                "Retweet Count": 10, 
                "Mention Count": 1, 
                "Follower Count": 500, 
                "Verified": False, 
                "Created At": "2023-10-27 10:00:00"
            },
            {
                "Tweet": "Click here to win $1000 free money now!!! #giveaway #free", 
                "Retweet Count": 500, 
                "Mention Count": 50, 
                "Follower Count": 10, 
                "Verified": False, 
                "Created At": "2023-10-27 11:00:00"
            }
        ]
        
        try:
            predictor = Predictor(model_path=model_save_path, scaler_path=scaler_save_path)
            predictions, probabilities = predictor.predict(new_data)
            
            print("\n--- Inference Results ---")
            for i, item in enumerate(new_data):
                label = "Bot" if predictions[i] == 1 else "Human"
                print(f"Data: {item[TEXT_FEATURE][:50]}... -> Prediction: {label} (Probability: {probabilities[i]:.4f})")
                
        except FileNotFoundError as e:
            print(e) # Should be caught by initial checks now
        except Exception as e:
            print(f"An error occurred during prediction: {e}")

