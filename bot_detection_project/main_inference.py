import argparse
import pandas as pd
import os
import json
from typing import List, Dict

from src.inference import Predictor
from src.config import MODEL_SAVE_DIR, RESULTS_DIR
from src.data_loader import SCALER_PATH

def load_data_from_file(file_path: str) -> List[Dict]:
    """Load data from a CSV or JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
        
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        # Convert DataFrame rows to list of dictionaries
        data = df.to_dict(orient="records")
    elif file_path.lower().endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .json file.")
        
    return data

def main(input_file: str, output_file: str = None):
    """
    Main function to run inference on input data.
    
    Args:
        input_file (str): Path to the input data file (CSV or JSON).
        output_file (str, optional): Path to save the predictions (CSV). Defaults to None (print to console).
    """
    print(f"Starting inference process for file: {input_file}")
    
    try:
        # Define paths for model and scaler
        model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
        scaler_path = SCALER_PATH
        
        # Check if model and scaler exist (basic check)
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Error: Model (", model_path, ") or scaler (", scaler_path, ") not found.")
            print("Please ensure pipeline.py has been run to create the scaler, and the model has been trained.")
            return
            
        # Load input data
        print(f"Loading data from {input_file}...")
        input_data = load_data_from_file(input_file)
        if not input_data:
            print("No data found in the input file.")
            return
            
        # Initialize predictor
        print("Initializing predictor...")
        predictor = Predictor(model_path=model_path, scaler_path=scaler_path)
        
        # Make predictions
        print("Making predictions...")
        predictions, probabilities = predictor.predict(input_data)
        
        # Prepare output
        output_data = []
        for i, item in enumerate(input_data):
            item["predicted_label"] = int(predictions[i]) # 0 for Human, 1 for Bot
            item["bot_probability"] = float(probabilities[i])
            output_data.append(item)
            
        # Output results
        if output_file:
            print(f"Saving predictions to {output_file}...")
            output_df = pd.DataFrame(output_data)
            # Ensure results directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_df.to_csv(output_file, index=False)
            print("Predictions saved successfully.")
        else:
            print("\n--- Inference Results ---")
            for item in output_data:
                label = "Bot" if item["predicted_label"] == 1 else "Human"
                # Display limited text to avoid clutter
                text_preview = item.get("Tweet", "N/A")[:50] + "..." if item.get("Tweet") else "N/A"
                print(f"Data: {text_preview} -> Prediction: {label} (Probability: {item["bot_probability"]:.4f})")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bot Detection Inference")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input data file (CSV or JSON). Must contain required features (e.g., Tweet, Retweet Count, etc.)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="(Optional) Path to save the predictions as a CSV file. If not provided, results are printed to console."
    )
    
    args = parser.parse_args()
    
    # Construct full output path if specified
    output_path = None
    if args.output:
        # Ensure output is saved in the results directory if only a filename is given
        if os.path.dirname(args.output) == "":
            output_path = os.path.join(RESULTS_DIR, args.output)
        else:
            output_path = args.output
            
    main(input_file=args.input, output_file=output_path)
