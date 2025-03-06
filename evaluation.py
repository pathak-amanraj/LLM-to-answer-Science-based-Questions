import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    cohen_kappa_score, 
    log_loss, 
    confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer

def calculate_average_precision(true_labels, predictions, k=None):
    """
    Calculate Average Precision for multi-class/multi-label classification
    
    Parameters:
    true_labels (list): Ground truth labels
    predictions (list): Predicted labels
    k (int, optional): Top-k predictions to consider
    
    Returns:
    float: Average Precision score
    """
    def compute_ap(true_label, pred_labels, k=None):
        if k is not None:
            pred_labels = pred_labels[:k]
        
        # Calculate precision at each relevant position
        precisions = []
        num_hits = 0
        
        for i, pred in enumerate(pred_labels):
            if pred == true_label:
                num_hits += 1
                precisions.append(num_hits / (i + 1))
        
        # If no hits, return 0
        if not precisions:
            return 0
        
        # Return mean of precisions
        return np.mean(precisions)
    
    # Calculate Average Precision for each sample
    aps = []
    for true, pred in zip(true_labels, predictions):
        ap = compute_ap(true, pred, k)
        aps.append(ap)
    
    # Return Mean Average Precision
    return np.mean(aps)

def evaluate_model_performance(true_labels, predictions):
    """
    Compute multiple evaluation metrics for multi-class classification
    
    Parameters:
    true_labels (list or array): Ground truth labels
    predictions (list or array): Predicted labels
    
    Returns:
    dict: Dictionary containing various performance metrics
    """
    # Prepare one-hot encoded labels
    mlb = MultiLabelBinarizer(classes=['A', 'B', 'C', 'D', 'E'])
    
    # Convert predictions and true labels to one-hot encoded format
    true_labels_encoded = mlb.fit_transform([[label] for label in true_labels])
    predictions_encoded = mlb.transform([[label] for label in predictions])
    
    # 1. F1 Score (Macro)
    f1 = f1_score(true_labels_encoded, predictions_encoded, average='macro')
    
    # 2. AUC-ROC (One-vs-Rest)
    try:
        auc_roc = roc_auc_score(true_labels_encoded, predictions_encoded, multi_class='ovr', average='macro')
    except ValueError:
        auc_roc = None
    
    # 3. Cohen's Kappa
    # For multi-label, we'll use the first predicted label
    first_pred_labels = [pred[0] if isinstance(pred, list) else pred for pred in predictions]
    first_true_labels = [true[0] if isinstance(true, list) else true for true in true_labels]
    
    kappa = cohen_kappa_score(first_true_labels, first_pred_labels)
    
    # 4. Logarithmic Loss
    # Probability prediction placeholder - you'll need to modify this
    prob_predictions = np.zeros((len(true_labels), 5))
    for i, pred in enumerate(predictions):
        for label in pred:
            prob_predictions[i, ord(label) - ord('A')] = 1/len(pred)
    
    # One-hot encode true labels for log loss
    true_labels_one_hot = np.zeros((len(true_labels), 5))
    for i, label in enumerate(true_labels):
        true_labels_one_hot[i, ord(label) - ord('A')] = 1
    
    logloss = log_loss(true_labels_one_hot, prob_predictions)
    
    # 5. Mean Average Precision (MAP)
    # Calculate MAP@1, MAP@3, MAP@5
    map_1 = calculate_average_precision(true_labels, predictions, k=1)
    map_3 = calculate_average_precision(true_labels, predictions, k=3)
    map_5 = calculate_average_precision(true_labels, predictions, k=5)
    
    # Confusion Matrix
    cm = confusion_matrix(first_true_labels, first_pred_labels, labels=['A', 'B', 'C', 'D', 'E'])
    
    return {
        'F1 Score (Macro)': f1,
        'AUC-ROC (One-vs-Rest)': auc_roc,
        'Cohen\'s Kappa': kappa,
        'Logarithmic Loss': logloss,
        'Mean Average Precision @1': map_1,
        'Mean Average Precision @3': map_3,
        'Mean Average Precision @5': map_5,
        'Confusion Matrix': cm
    }

# Visualization and reporting functions (from previous example)
# ... (keep the previous visualization and reporting code)

# Example usage
# ground_truth_labels = df_valid['answer'].tolist()  # Replace with actual ground truth column
# performance_metrics = evaluate_model_performance(ground_truth_labels, predictions)

# Detailed explanation of Mean Average Precision (MAP)
"""
Mean Average Precision (MAP) Explanation:

1. What is MAP?
   - A metric that evaluates the ranking quality of predictions
   - Calculates precision at different ranks and averages them
   - Useful for multi-label or ranked prediction problems

2. MAP Variants:
   - MAP@1: Precision at top 1 prediction
   - MAP@3: Precision at top 3 predictions
   - MAP@5: Precision at top 5 predictions

3. How it works:
   - For each sample, calculate Average Precision (AP)
   - AP measures how well the correct label is ranked
   - Mean of APs across all samples gives MAP

4. Interpretation:
   - Higher MAP indicates better ranking of predictions
   - Ranges from 0 to 1
   - Considers both precision and ranking
"""

# Optional: Save detailed MAP explanation
with open('map_explanation.txt', 'w') as f:
    f.write("""Mean Average Precision (MAP) Explanation:

1. What is MAP?
   - A metric that evaluates the ranking quality of predictions
   - Calculates precision at different ranks and averages them
   - Useful for multi-label or ranked prediction problems

2. MAP Variants:
   - MAP@1: Precision at top 1 prediction
   - MAP@3: Precision at top 3 predictions
   - MAP@5: Precision at top 5 predictions

3. How it works:
   - For each sample, calculate Average Precision (AP)
   - AP measures how well the correct label is ranked
   - Mean of APs across all samples gives MAP

4. Interpretation:
   - Higher MAP indicates better ranking of predictions
   - Ranges from 0 to 1
   - Considers both precision and ranking
""")