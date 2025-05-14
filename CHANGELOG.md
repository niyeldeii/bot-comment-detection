# Changelog

## [YYYY-MM-DD] - Model Architecture Fix

### Issue Identified

The original model implementation in `src/models/bert_model.py` utilized the `pooler_output` from the BERT model as the primary text representation for the downstream classification layers. The training logs indicated that the model was failing to learn effectively, with validation loss stagnating near log(2) and metrics like precision/recall/F1 dropping to zero in some epochs. This suggested the model was collapsing to predict only the majority class.

While the `pooler_output` is derived from the `[CLS]` token's hidden state, it is primarily trained for the Next Sentence Prediction (NSP) task during BERT's pre-training. For many downstream classification tasks, it has been observed that this output might not be the most informative representation.

### Fix Implemented

The `forward` method within the `BERTBotDetector` class in `src/models/bert_model.py` was modified to use the `last_hidden_state` of the `[CLS]` token directly. This is generally considered a more robust feature representation for sentence-level classification tasks.

**Change:**

```diff
--- a/src/models/bert_model.py
+++ b/src/models/bert_model.py
@@ -125,8 +125,9 @@
             return_dict=True
         )
 
-        # Process each feature type separately
-        bert_features = self.bert_processor(bert_output.pooler_output)
+        # Use the [CLS] token's last hidden state for better classification performance
+        cls_hidden_state = bert_output.last_hidden_state[:, 0]
+        bert_features = self.bert_processor(cls_hidden_state)
         num_features = self.numerical_processor(numerical_features)
         bool_features = self.boolean_processor(boolean_features)
 

```

### Validation

The `pipeline.py` script was executed successfully after this change. The pipeline completed all steps up to the point of initiating training, including data loading, preprocessing, model initialization (with the corrected architecture), and trainer setup. This confirms the code is syntactically correct and ready for the user to commence training.

This change is expected to improve the model's ability to learn discriminative features from the text, leading to better performance during training and evaluation.
