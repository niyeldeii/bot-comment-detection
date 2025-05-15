# Bot Detection Model using PyTorch

This project implements a bot detection model using PyTorch, designed to identify bot-like behavior in comments or posts on web platforms. It utilizes a combination of text features (from the comment/post content) and behavioral features (like retweet count, follower count, verification status, etc.) to make predictions.

The default model is a fine-tuned BERT model, but the framework also includes implementations for LSTM with Attention and a Logistic Regression baseline, allowing for easy swapping.

## Project Structure

```
.bot_detection_project/
├── data/
│   └── labeled_twitter_data.csv    # Dataset
├── models/                         # Directory for saved model checkpoints (e.g., best_model_*.pth from training runs)
├── results/                        # Directory for evaluation results from training runs
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
│   │   ├── lstm_model.py           # LSTM-based model implementation (optional)
│   │   └── logistic_regression.py  # Logistic Regression baseline implementation (optional)
│   ├── trainer.py                  # Training and validation loop logic, EarlyStopping, FocalLoss
│   ├── evaluator.py                # Evaluation logic for the test set (used by trainer)
│   └── utils.py                    # Utility functions (e.g., set_seed)
├── pipeline.py                     # Main script to run the data processing, training, and evaluation
├── main_inference.py               # Dedicated script for running inference on new data
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── performance_report_template.md  # Placeholder for performance report
```

## Features

*   **Data Preprocessing**: Handles text cleaning (e.g., replacing URLs, user mentions, hashtags with special tokens), feature extraction (time features from timestamps), filling missing numerical values, and data splitting (train/validation/test). The `StandardScaler` for numerical features is fitted on the training set and applied consistently.
*   **Multimodal Input**: Models are designed to accept both text embeddings (from BERT or LSTM) and additional numerical/boolean features.
*   **Multiple Model Architectures**: 
    *   BERT (fine-tuned `roberta-large` by default, with a custom head optimized for the task).
    *   LSTM (optional, defined in `src/models/lstm_model.py`).
    *   Logistic Regression (optional baseline, defined in `src/models/logistic_regression.py`).
*   **Training Setup**: Includes standard training components:
    *   Loss Function: `torch.nn.CrossEntropyLoss` is used by default. Class weights can be applied to handle imbalance if `USE_CLASS_WEIGHTS` is enabled in `config.py`. `FocalLoss` is also defined in `trainer.py` and can be integrated if needed.
    *   Optimizer: AdamW (default for BERT), with configurable differential learning rates for BERT base vs. custom head.
    *   Learning Rate Scheduler: `get_linear_schedule_with_warmup` (linear decay with warmup) and `ReduceLROnPlateau` are available and configurable via `SCHEDULER_TYPE` in `config.py`.
    *   Early Stopping: Monitors validation loss and saves the best model checkpoint.
    *   Gradient Accumulation & Clipping: Supported for stable training of large models.
*   **Evaluation**: Calculates Precision, Recall, F1-Score, and ROC-AUC. Generates classification reports and confusion matrices. Results are saved to the `results/` directory from the main training pipeline.
*   **Inference**: Uses `main_inference.py` for making predictions on new data with a trained model.
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
6.  **Prepare Data**: Ensure your dataset (e.g., `labeled_twitter_data.csv`) is in the `data/` directory. Update `DATA_PATH` in `src/config.py` if your file has a different name or location.

7.  **Run Training Pipeline**: Execute the main pipeline script. By default, this will use the BERT model.
    ```bash
    !python pipeline.py --model bert 
    ```
    *   Training will commence, showing epoch progress, loss, and validation metrics. 
    *   Model checkpoints (including the best model based on validation performance) and experiment configurations will be saved to a timestamped subdirectory within `./models/` (e.g., `./models/roberta-large_YYYYMMDD_HHMMSS/`).
    *   Evaluation results from the test set run at the end of training will be saved in the `results/` directory associated with the `ModelTrainer` (usually within the experiment-specific model save directory).

8.  **Run Inference**: After training, you can use the `main_inference.py` script to make predictions on new data. You will need to specify the path to your trained model checkpoint.
    ```bash
    # Example (ensure you have a trained model saved, e.g., models/roberta-large_YYYYMMDD_HHMMSS/best_model.pth)
    !python main_inference.py --model_path path/to/your/best_model.pth --input path/to/new_data.csv --output predictions.csv
    ```
    *   The input file should be a CSV containing the same features as the training data.
    *   If no output path is specified, results might be printed to the console or handled as per the script's logic.

## Configuration

Key parameters can be adjusted in `src/config.py`:

*   `DATA_PATH`: Path to the dataset.
*   `MODEL_SAVE_DIR`, `RESULTS_DIR`: Base directories for saving models and results.
*   `MODEL_NAME`: HuggingFace model name (e.g., "roberta-large").
*   `MAX_LEN`, `BATCH_SIZE`, `EPOCHS`: Core training hyperparameters.
*   `LEARNING_RATE`: Learning rate for the classifier head/non-BERT parts of the model.
*   `BERT_LR`: Specific learning rate for the BERT base model layers (enables differential learning).
*   `WARMUP_STEPS`: Number of warmup steps for the linear learning rate scheduler.
*   `SCHEDULER_TYPE`: Type of LR scheduler ('linear' or 'ReduceLROnPlateau').
*   `TEXT_FEATURE`, `NUMERICAL_FEATURES`, `BOOLEAN_FEATURES`, `TARGET_FEATURE`: Column names in the dataset.
*   `USE_OVERSAMPLING`, `USE_CLASS_WEIGHTS`: Flags to enable/disable techniques for handling class imbalance.
*   `DROPOUT_RATE`: Dropout probability for regularization in the custom model head.
*   `WEIGHT_DECAY`: L2 regularization strength for the optimizer.
*   `GRAD_CLIP`: Gradient clipping value.
*   `GRAD_ACCUMULATION_STEPS`: Number of steps to accumulate gradients before an optimizer update.
*   `FREEZE_BERT_LAYERS`: Number of initial BERT layers to freeze (set to `0` to train all BERT layers).

## Key Configuration Choices & Rationale

The current default configuration in `src/config.py` is optimized for robust training and performance with the `roberta-large` model on this type of task. Key choices include:

*   **`MODEL_NAME = "roberta-large"`**: Utilizes a powerful pre-trained transformer known for strong performance on various NLP tasks.
*   **`EPOCHS = 20`**: Provides sufficient iterations for the model to converge, especially when fine-tuning a large model.
*   **`MAX_LEN = 128`**: Optimized sequence length based on the nature of tweets (typically short).
*   **`BATCH_SIZE = 8`**: A smaller batch size, often necessary for large models like RoBERTa-large to fit within typical GPU memory, used in conjunction with gradient accumulation if needed.
*   **`BERT_LR = 2e-5`, `LEARNING_RATE = 5e-4`**: Employs differential learning rates. A smaller learning rate for the pre-trained BERT layers helps in stable fine-tuning, while a potentially larger rate for the custom classifier head allows it to adapt more quickly.
*   **`FREEZE_BERT_LAYERS = 0`**: All layers of the RoBERTa model are trainable by default. This allows the model to adapt maximally to the specific dataset and task, which can be beneficial if computational resources allow and overfitting is managed (e.g., via dropout, weight decay, early stopping).
*   **`DROPOUT_RATE = 0.3`**: Moderate dropout in the classifier head to help prevent overfitting.
*   **Early Stopping & Schedulers**: Robust mechanisms (`EarlyStopping`, linear LR scheduler with warmup by default) are in place to manage training dynamics, prevent overfitting, and find optimal model state.
*   **Optimized Model Architecture**: The `BERTBotDetector` in `src/models/bert_model.py` uses a simplified custom head (linear layers with LayerNorm and ReLU) with a hidden size of 128, designed to efficiently combine text, numerical, and boolean features for this specific bot detection task.

These settings provide a strong baseline. Depending on your specific dataset and computational resources, you might experiment with these values.

## Notes

*   Running `python pipeline.py` will execute the full training and evaluation pipeline by default using the model specified by the `--model` argument (BERT by default).
*   Ensure you have sufficient resources (GPU with adequate VRAM recommended for RoBERTa-large) for training.
*   The provided `labeled_twitter_data.csv` is a sample. For best results, use your own larger and representative dataset.
*   The LSTM and Logistic Regression models are provided as alternatives and may require specific tuning or adjustments to their configurations or feature handling to achieve optimal performance compared to the default BERT setup.
