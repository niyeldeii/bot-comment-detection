import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, Tuple

from src.config import DEVICE, RESULTS_DIR
from src.models.bert_model import BERTBotDetector # Assuming BERT is the default
# Import other models if needed for evaluation flexibility

class ModelEvaluator:
    """
    Class to handle the evaluation of the trained model on the test set
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_path: str # Path to the saved model state dictionary
    ):
        """
        Initialize the ModelEvaluator
        
        Args:
            model: The PyTorch model architecture (instantiated)
            test_loader: DataLoader for the test set
            model_path: Path to the saved model weights (.pth file)
        """
        self.model = model
        self.test_loader = test_loader
        self.model_path = model_path
        self.model.to(DEVICE)
        
    def load_model(self):
        """Load the trained model weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}. Please train the model first.")
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        print(f"Model loaded from {self.model_path}")
        
    def evaluate(self) -> Tuple[Dict[str, float], str, np.ndarray]:
        """
        Evaluate the model on the test set and return metrics.
        
        Returns:
            Tuple containing:
            - Dictionary of performance metrics (precision, recall, f1, roc_auc)
            - Classification report string
            - Confusion matrix numpy array
        """
        self.load_model()
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                numerical_features = batch["numerical_features"].to(DEVICE)
                boolean_features = batch["boolean_features"].to(DEVICE)
                # Test loader might not have targets, handle this if necessary
                # Assuming test loader provides targets for evaluation purposes here
                # If not, this part needs adjustment based on how test data is prepared
                if "targets" in batch:
                    targets = batch["targets"].to(DEVICE)
                    all_targets.extend(targets.cpu().numpy())
                
                # Forward pass - Adapt if using Logistic Regression
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    boolean_features=boolean_features
                )
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of positive class (bot)

        if not all_targets:
             print("Warning: No targets found in the test loader. Cannot calculate performance metrics.")
             return {}, "No targets provided for evaluation.", np.array([])

        # Calculate metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_targets, all_probs)
        except ValueError: # Handle cases where only one class is present in targets
             roc_auc = 0.0
             print("Warning: ROC AUC could not be calculated. Only one class present in test targets.")

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        
        # Generate classification report and confusion matrix
        report = classification_report(all_targets, all_preds, target_names=["Human", "Bot"], zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)
        
        print("\nTest Set Evaluation Results:")
        print(f"  Precision: {metrics["precision"]:.4f}")
        print(f"  Recall: {metrics["recall"]:.4f}")
        print(f"  F1 Score: {metrics["f1_score"]:.4f}")
        print(f"  ROC AUC: {metrics["roc_auc"]:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
        
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)
        with open(report_path, "w") as f:
            f.write(report)
        pd.DataFrame(cm, index=["Actual Human", "Actual Bot"], columns=["Predicted Human", "Predicted Bot"]).to_csv(cm_path)
        
        print(f"\nEvaluation results saved to: {RESULTS_DIR}")
        
        return metrics, report, cm

# Note: This script defines the Evaluator class.
# It will be called after training is complete (likely manually by the user or via a separate script).
# The main pipeline.py focuses on setting up for training.
