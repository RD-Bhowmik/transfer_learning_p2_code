import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    precision_recall_curve, average_precision_score,
    f1_score, matthews_corrcoef, cohen_kappa_score
)
import traceback
import pandas as pd
from tabulate import tabulate
from tensorflow.keras import losses, backend as K
import tensorflow as tf

def calculate_advanced_metrics(y_true, y_pred, y_pred_prob):
    """Calculate advanced performance metrics"""
    return {
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'average_precision': float(average_precision_score(y_true, y_pred_prob))
    }

def plot_metrics_over_time(tracking_dict, save_folder):
    """Plot metrics evolution over training iterations"""
    metrics_to_plot = {
        'Validation Accuracy': 'val_accuracies',
        'Validation Loss': 'val_losses',
        'Precision': 'precision',
        'Recall': 'recall',
        'AUC': 'auc'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        if metric_key in tracking_dict and tracking_dict[metric_key]:
            axes[idx].plot(tracking_dict[metric_key], marker='o')
            axes[idx].set_title(f'{metric_name} Over Time')
            axes[idx].set_xlabel('Iteration')
            axes[idx].set_ylabel(metric_name)
            axes[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'metrics_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, save_folder):
    """Plot confusion matrix as a heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_and_pr_curves(y_true, y_pred_prob, save_folder):
    """Plot ROC and Precision-Recall curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    ax2.plot(recall, precision)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve (AP={avg_precision:.3f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_distribution(y_pred_prob, y_true, save_folder):
    """Plot probability distribution for predictions"""
    plt.figure(figsize=(10, 6))
    for label in [0, 1]:
        mask = y_true == label
        plt.hist(y_pred_prob[mask], bins=20, alpha=0.5, 
                label=f'Class {label}', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution by Class')
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_metrics(y_test, y_pred, y_pred_prob, save_folder):
    """Save detailed performance metrics including classification report, confusion matrix and visualizations"""
    try:
        # Create metrics folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        
        # Calculate advanced metrics
        advanced_metrics = calculate_advanced_metrics(y_test, y_pred, y_pred_prob)
        
        # Calculate and save numerical metrics
        detailed_metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'probabilities_summary': {
                'mean': float(np.mean(y_pred_prob)),
                'std': float(np.std(y_pred_prob)),
                'min': float(np.min(y_pred_prob)),
                'max': float(np.max(y_pred_prob))
            },
            'advanced_metrics': advanced_metrics
        }
        
        # Save metrics to JSON
        metrics_path = os.path.join(save_folder, 'detailed_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
        
        # Save text report
        report_path = os.path.join(save_folder, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\n\nProbabilities Summary:\n")
            f.write(f"Mean: {np.mean(y_pred_prob):.4f}\n")
            f.write(f"Std: {np.std(y_pred_prob):.4f}\n")
            f.write(f"Min: {np.min(y_pred_prob):.4f}\n")
            f.write(f"Max: {np.max(y_pred_prob):.4f}\n")
        
        # Generate visualizations
        plot_confusion_matrix_heatmap(y_test, y_pred, save_folder)
        plot_roc_and_pr_curves(y_test, y_pred_prob, save_folder)
        plot_probability_distribution(y_pred_prob, y_test, save_folder)
        
        print(f"Detailed metrics and visualizations saved to: {save_folder}")
        return True
        
    except Exception as e:
        print(f"Error saving detailed metrics: {str(e)}")
        return False

def track_hyperparameter_performance():
    """Track performance of each hyperparameter combination"""
    return {
        'learning_rates': [],
        'batch_sizes': [],
        'dropout_rates': [],
        'val_accuracies': [],
        'val_losses': [],
        'training_times': [],
        'epochs_trained': [],
        'precision': [],
        'recall': [],
        'auc': []
    }

def generate_performance_report(tracking_dict, viz_folder):
    """Generate comprehensive performance report across iterations"""
    report = {
        'overall_improvement': {
            'val_accuracies': tracking_dict['val_accuracies'],
            'val_losses': tracking_dict['val_losses'],
            'training_times': tracking_dict['training_times'],
            'initial_accuracy': tracking_dict['val_accuracies'][0] if tracking_dict['val_accuracies'] else None,
            'final_accuracy': tracking_dict['val_accuracies'][-1] if tracking_dict['val_accuracies'] else None,
            'total_iterations': len(tracking_dict['val_accuracies'])
        },
        'best_performance': {
            'best_accuracy': max(tracking_dict['val_accuracies']) if tracking_dict['val_accuracies'] else None,
            'best_iteration': tracking_dict['val_accuracies'].index(max(tracking_dict['val_accuracies'])) if tracking_dict['val_accuracies'] else None,
            'corresponding_hyperparameters': {
                'learning_rate': tracking_dict['learning_rates'][tracking_dict['val_accuracies'].index(max(tracking_dict['val_accuracies']))] if tracking_dict['val_accuracies'] else None,
                'batch_size': tracking_dict['batch_sizes'][tracking_dict['val_accuracies'].index(max(tracking_dict['val_accuracies']))] if tracking_dict['val_accuracies'] else None,
                'dropout_rate': tracking_dict['dropout_rates'][tracking_dict['val_accuracies'].index(max(tracking_dict['val_accuracies']))] if tracking_dict['val_accuracies'] else None
            }
        },
        'metrics_summary': {
            'precision': {
                'max': max(tracking_dict['precision']) if tracking_dict['precision'] else None,
                'mean': sum(tracking_dict['precision'])/len(tracking_dict['precision']) if tracking_dict['precision'] else None
            },
            'recall': {
                'max': max(tracking_dict['recall']) if tracking_dict['recall'] else None,
                'mean': sum(tracking_dict['recall'])/len(tracking_dict['recall']) if tracking_dict['recall'] else None
            },
            'auc': {
                'max': max(tracking_dict['auc']) if tracking_dict['auc'] else None,
                'mean': sum(tracking_dict['auc'])/len(tracking_dict['auc']) if tracking_dict['auc'] else None
            }
        }
    }
    
    # Save report
    report_path = os.path.join(viz_folder, "model_evolution", "performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

def analyze_feature_importance(model, X_test, meta_X_test, clinical_features, viz_folder):
    """Analyze and visualize feature importance"""
    try:
        # Get base prediction
        base_pred = model.predict([np.expand_dims(X_test[0], 0), 
                                 np.expand_dims(meta_X_test[0], 0)])[0][0]
        
        # Calculate feature importance
        importance_dict = {}
        for i, feature in enumerate(clinical_features):
            # Create modified metadata
            modified_meta = meta_X_test[0].copy()
            modified_meta[i] = 0  # Zero out feature
            
            # Get new prediction
            new_pred = model.predict([np.expand_dims(X_test[0], 0), 
                                    np.expand_dims(modified_meta, 0)])[0][0]
            
            # Calculate importance (convert to Python float)
            importance = float(abs(base_pred - new_pred))
            importance_dict[feature] = importance
        
        # Save importance scores
        save_path = os.path.join(viz_folder, 'feature_importance.json')
        with open(save_path, 'w') as f:
            json.dump(importance_dict, f, indent=4)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        pos = np.arange(len(features))
        
        plt.barh(pos, [importances[i] for i in sorted_idx])
        plt.yticks(pos, [features[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Clinical Feature Importance Analysis')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'feature_importance.png'))
        plt.close()
        
        return importance_dict
        
    except Exception as e:
        print(f"Error analyzing feature importance: {str(e)}")
        traceback.print_exc()
        return {}

def find_optimal_threshold(y_true, y_pred_prob):
    """Find optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def generate_model_report(model, history, test_data, output_folder, model_name):
    """Generate comprehensive model performance report"""
    # Evaluate model
    test_metrics = model.evaluate(test_data)
    
    # Create metrics dictionary
    report = {
        'model_name': model_name,
        'training_metrics': {
            'final_train_acc': history.history['accuracy'][-1],
            'final_val_acc': history.history['val_accuracy'][-1],
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        },
        'test_metrics': dict(zip(model.metrics_names, test_metrics))
    }
    
    # Generate classification report and confusion matrix
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    y_pred = model.predict(test_data)
    y_pred_prob = y_pred[:, 0]  # Assuming binary classification
    
    # After existing predictions
    optimal_threshold = find_optimal_threshold(y_true, y_pred_prob)
    
    # Update predictions with optimal threshold
    y_pred_class = (y_pred_prob > optimal_threshold).astype(int)
    
    # Save threshold value
    report['optimal_threshold'] = float(optimal_threshold)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred_class, 
        target_names=['Class 0', 'Class 1'],
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    
    # Save text report
    with open(os.path.join(output_folder, 'performance_report.txt'), 'w') as f:
        # Basic info
        f.write(f"{model_name} Performance Report\n{'='*50}\n")
        f.write(f"Model Architecture: {model.name}\n")
        f.write(f"Input Shape: {model.input_shape}\n\n")
        
        # Training metrics
        f.write("Training Metrics:\n")
        f.write(f"- Final Training Accuracy: {report['training_metrics']['final_train_acc']:.4f}\n")
        f.write(f"- Final Validation Accuracy: {report['training_metrics']['final_val_acc']:.4f}\n")
        f.write(f"- Final Training Loss: {report['training_metrics']['final_train_loss']:.4f}\n")
        f.write(f"- Final Validation Loss: {report['training_metrics']['final_val_loss']:.4f}\n\n")
        
        # Test metrics
        f.write("Test Metrics:\n")
        for k, v in report['test_metrics'].items():
            f.write(f"- {k}: {v:.4f}\n")
        
        # Classification report
        f.write("\nClassification Report:\n")
        f.write(tabulate(
            pd.DataFrame(class_report).transpose().round(2),
            headers='keys',
            tablefmt='psql'
        ))
        
        # Confusion matrix
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm, formatter={'int': lambda x: f"{x:3d}"}))
        
        # Probabilities summary
        f.write("\n\nProbabilities Summary:\n")
        f.write(f"Mean: {y_pred.mean():.4f}\n")
        f.write(f"Std: {y_pred.std():.4f}\n")
        f.write(f"Min: {y_pred.min():.4f}\n")
        f.write(f"Max: {y_pred.max():.4f}\n")
    
    # Save visualizations
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_metrics.png'))
    plt.close()
    
    # Confusion matrix plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()
    
    # Probability distribution plot
    plt.figure()
    sns.histplot(y_pred, bins=20, kde=True)
    plt.title('Predicted Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_folder, 'probability_distribution.png'))
    plt.close()

def create_model_comparison(histories, output_folder):
    """Create comparison plot of all models"""
    plt.figure(figsize=(15, 6))
    
    # Filter models
    allowed_models = {'B0', 'B3', 'B4', 'B5'}
    histories = {k:v for k,v in histories.items() if k in allowed_models}
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    for version, history in histories.items():
        plt.plot(history['val_accuracy'], label=f'EfficientNet{version}')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss comparison
    plt.subplot(1, 2, 2)
    for version, history in histories.items():
        plt.plot(history['val_loss'], label=f'EfficientNet{version}')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'model_comparison.png'))
    plt.close()

class WeightedBCE(losses.Loss):
    def __init__(self, neg_weight=2.0, pos_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.neg_weight = neg_weight
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        # Reshape y_true to match y_pred shape
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        y_true = tf.cast(y_true, tf.float32)  # Convert labels to float32
        
        bce = losses.binary_crossentropy(y_true, y_pred)
        weights = y_true * self.pos_weight + (1 - y_true) * self.neg_weight
        return K.mean(weights * bce)

def specificity(y_true, y_pred):
    """Custom specificity metric"""
    # Cast both inputs to float32 to ensure type consistency
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    tn = K.sum(K.round(1 - y_true) * K.round(1 - y_pred))
    fp = K.sum(K.round(1 - y_true) * K.round(y_pred))
    return tn / (tn + fp + K.epsilon())

if __name__ == "__main__":
    print("Testing metrics module...")
    
    # Create sample data
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pred_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.6])
    
    # Test folder
    test_folder = "test_metrics"
    
    # Save metrics and generate visualizations
    success = save_detailed_metrics(y_true, y_pred, y_pred_prob, test_folder)
    print(f"\nMetrics and visualizations saved successfully: {success}")
    
    # Test tracking dict visualization
    tracking_dict = {
        'val_accuracies': [0.7, 0.75, 0.8, 0.82, 0.85],
        'val_losses': [0.5, 0.45, 0.4, 0.35, 0.3],
        'precision': [0.72, 0.76, 0.81, 0.83, 0.86],
        'recall': [0.68, 0.73, 0.79, 0.81, 0.84],
        'auc': [0.75, 0.78, 0.82, 0.84, 0.87]
    }
    plot_metrics_over_time(tracking_dict, test_folder) 