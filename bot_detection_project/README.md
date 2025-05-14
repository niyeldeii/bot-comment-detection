# Bot Detection Model using PyTorch

This project implements a bot detection model using PyTorch, designed to identify bot-like behavior in comments or posts on web platforms. It utilizes a combination of text features (from the comment/post content) and behavioral features (like retweet count, follower count, verification status, etc.) to make predictions.

The default model is a fine-tuned BERT model, but the framework also includes implementations for LSTM with Attention and a Logistic Regression baseline, allowing for easy swapping.

## Project Structure

```
.bot_detection_project/
├── data/
│   └── labeled_twitter_data.csv    # Dataset
├── models/                         # Directory for saved model checkpoints (.pth files)
│   ├── best_model.pth              # Example placeholder for the best model after training
│   └── scaler.joblib               # Saved feature scaler for consistent preprocessing
├── results/                        # Directory for evaluation results
│   ├── evaluation_results.json
│   ├── classification_report.txt
│   └── confusion_matrix.csv
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration settings (paths, hyperparameters)
│   ├── data_loader.py              # Data loading, preprocessing, and Dataset/DataLoader classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bert_model.py           # BERT-based model implementation
│   │   ├── lstm_model.py           # LSTM-based model implementation
│   │   └── logistic_regression.py  # Logistic Regression baseline implementation
│   ├── trainer.py                  # Training and validation loop logic, EarlyStopping
│   ├── evaluator.py                # Evaluation logic for the test set
│   ├── inference.py                # Script/class for making predictions on new data
│   └── utils.py                    # Utility functions (e.g., set_seed)
├── pipeline.py                     # Main script to run the data processing and training setup
├── main_inference.py               # Dedicated script for running inference on new data
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── performance_report_template.md  # Placeholder for performance report
```

## Features

*   **Data Preprocessing**: Handles text cleaning (lowercasing, removing URLs/mentions/special chars), feature extraction (time features from timestamps), missing value imputation (for hashtags), and data splitting (train/validation/test).
*   **Multimodal Input**: Models are designed to accept both text embeddings (from BERT or LSTM) and additional numerical/boolean features.
*   **Multiple Model Architectures**: 
    *   BERT (fine-tuned, default)
    *   LSTM with Attention (optional)
    *   Logistic Regression (optional baseline)
*   **Training Setup**: Includes standard training components:
    *   Loss Function: CrossEntropyLoss (with automatic class weighting for imbalance).
    *   Optimizer: AdamW (for BERT), Adam (configurable).
    *   Learning Rate Scheduler: Linear warmup (configurable).
    *   Early Stopping: Monitors validation loss and saves the best model.
*   **Evaluation**: Calculates Precision, Recall, F1-Score, and ROC-AUC. Generates classification reports and confusion matrices.
*   **Inference**: Provides a `Predictor` class in `inference.py` and a dedicated `main_inference.py` script for making predictions on new data.
*   **Colab Ready**: Designed for easy setup and execution in Google Colab.

## Setup and Usage (Google Colab)

1.  **Upload Project**: Upload the provided `bot_detection_project.zip` file to your Google Drive.
2.  **Mount Drive**: Mount your Google Drive in Colab:
    ```python
    from google.colab import drive
    drive.mount("/content/drive")
    ```
3.  **Unzip Project**: Unzip the project file to your Colab environment. Replace `/content/drive/MyDrive/path/to/bot_detection_project.zip` with the actual path to the zip file in your Drive.
    ```bash
    !unzip /content/drive/MyDrive/path/to/bot_detection_project.zip -d /content/
    ```
4.  **Navigate to Project Directory**:
    ```python
    import os
    os.chdir("/content/bot_detection_project")
    ```
5.  **Install Dependencies**: Install the required Python packages.
    ```bash
    !pip install -r requirements.txt
    ```
    *Note: This might take a few minutes, especially for PyTorch and Transformers.*

6.  **Run Pipeline (Setup Only)**: Execute the main pipeline script. This will perform data loading, preprocessing, model initialization, and prepare everything for training. It will *not* start the actual training yet.
    ```bash
    !python pipeline.py --model bert 
    ```
    *   You can change `--model bert` to `--model lstm` or `--model logistic_regression` to initialize a different model type (though note the warning in `pipeline.py` regarding Logistic Regression compatibility with the current trainer setup without modifications).

7.  **Start Training (Optional)**: To actually train the model, you need to:
    *   Edit the `pipeline.py` file.
    *   Find the line `# history = trainer.train()` near the end of the `main` function.
    *   Uncomment this line.
    *   Save the file.
    *   Re-run the pipeline script:
        ```bash
        !python pipeline.py --model bert 
        ```
    *   Training will commence, showing epoch progress, loss, and validation metrics. The best model based on validation loss will be saved to the `./models/` directory (e.g., `best_model.pth`).

8.  **Run Inference**: After training, you can use the dedicated inference script to make predictions on new data:
    ```bash
    !python main_inference.py --input path/to/new_data.csv --output predictions.csv
    ```
    *   The input file should be a CSV or JSON file containing the same features as the training data.
    *   If no output path is specified, results will be printed to the console.
    *   Example:
        ```bash
        !python main_inference.py --input data/test_samples.csv
        ```

## Configuration

Key parameters can be adjusted in `src/config.py`:

*   `DATA_PATH`: Path to the dataset.
*   `MODEL_SAVE_DIR`, `RESULTS_DIR`: Directories for saving models and results.
*   `MODEL_NAME`: HuggingFace model name for BERT.
*   `MAX_LEN`, `BATCH_SIZE`, `EPOCHS`: Training hyperparameters.
*   `LEARNING_RATE`: Learning rate for the *classifier head*.
*   `BERT_LR`: Learning rate for the *BERT layers* (used for differential learning).
*   `WARMUP_STEPS`: Number of warmup steps for the learning rate scheduler.
*   `TEXT_FEATURE`, `NUMERICAL_FEATURES`, `BOOLEAN_FEATURES`, `TARGET_FEATURE`: Column names in the dataset.
*   `USE_OVERSAMPLING`: Set to `True` to enable Random Oversampling (currently `False`).
*   `DROPOUT_RATE`: Dropout probability for regularization.
*   `WEIGHT_DECAY`: L2 regularization strength for the optimizer.
*   `GRAD_CLIP`: Gradient clipping value to prevent exploding gradients.
*   `FREEZE_BERT_LAYERS`: Number of initial BERT layers to freeze during training.

## Optimizations (Definitive Fix - Attempt 4)

Given the persistent failure of the model to learn effectively (ROC AUC ~0.5) despite previous attempts, a thorough root cause analysis was performed. The likely issues identified were **feature scaling/combination mismatch** and **gradient flow problems** during fine-tuning. The following definitive fixes were implemented:

1.  **Layer Normalization**: Added a `torch.nn.LayerNorm` layer in `bert_model.py` immediately after concatenating the BERT `pooler_output`, numerical features, and boolean features. This normalizes the combined features *before* they enter the intermediate MLP, addressing the potential mismatch in scales and distributions and stabilizing the learning of feature interactions.
2.  **Reduced Frozen Layers**: Reduced `FREEZE_BERT_LAYERS` from 8 to 6 in `config.py` to allow slightly more gradient flow into the pre-trained BERT model, potentially improving fine-tuning on relevant layers while maintaining stability.
3.  **Adjusted Differential Learning Rates**: Fine-tuned the learning rates in `config.py` (`BERT_LR = 3e-5`, `LEARNING_RATE = 7e-5`) to better suit the architecture with layer normalization and fewer frozen layers.
4.  **Increased Epochs**: Increased `EPOCHS` to 12 in `config.py` to provide more opportunity for the model to converge with the new architecture and training setup.

These changes directly target the suspected root causes and represent a robust strategy to enable effective learning.

## Notes

*   The project is set up *not* to train by default. You must uncomment the training call in `pipeline.py` to run training.
*   Ensure you have sufficient resources (RAM, GPU recommended for faster training) in your Colab environment, especially when using BERT.
*   The provided `labeled_twitter_data.csv` is used for demonstration. Replace or augment with your own data as needed.
*   The LSTM and Logistic Regression models might require adjustments (e.g., different tokenization/embedding strategies for LSTM, feature combination logic for LR) for optimal performance compared to the BERT setup.
