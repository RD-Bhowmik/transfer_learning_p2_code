import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import traceback
import json
import datetime
import itertools
import time
from tensorflow.keras.optimizers import Adam
from src.metrics import WeightedBCE, specificity, create_model_comparison

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import SAVE_FOLDER
from src.data_processing import (
    load_and_preprocess_data, align_data, preprocess_metadata,
    prepare_data_splits, setup_data_augmentation, create_augmentation_dataset,
    create_class_balanced_generator
)
from src.model import (
    create_model, setup_callbacks, train_model, evaluate_model,
    make_predictions, save_intermediate_model, load_previous_best_model,
    create_effnet_model, EFFNET_MODELS
)
from src.visualization import (
    visualize_results, visualize_metadata, visualize_model_performance,
    visualize_learning_dynamics, visualize_metadata_distributions,
    visualize_weight_updates, visualize_hyperparameter_effects,
    visualize_data_augmentation, visualize_clinical_patterns,
    visualize_model_interpretation
)
from src.metrics import (
    track_hyperparameter_performance, save_detailed_metrics,
    generate_performance_report, analyze_feature_importance,
    generate_model_report
)
from src.utils import (
    setup_visualization_folder, save_json, create_experiment_folder,
    log_training_info
)
from src.statistical_analysis import perform_statistical_tests, save_statistical_analysis
from src.model_interpretation import (
    ModelInterpreter, 
    safe_compare_model_weights
)
from src.patient_analysis import PatientAnalyzer
from src.advanced_visualization import (
    visualize_statistical_results, visualize_model_interpretations,
    visualize_patient_profiles, visualize_combined_analysis
)
from src.medical_preprocessing import ColposcopyPreprocessor, LesionAnalyzer
from src.clinical_metrics import ClinicalMetrics, ClinicalValidator
from src.clinical_reporting import ClinicalReport
from src.safety_monitoring import SafetyMonitor
from src.json_structure_visualizer import create_json_structure_visualization
from src.models.effnet_b3 import create_model as create_b3, load_and_preprocess_data as load_data_b3
from src.models.effnet_b4 import create_model as create_b4, load_and_preprocess_data as load_data_b4
from src.models.effnet_b0 import create_model as create_b0, load_and_preprocess_data as load_data_b0
from src.models.effnet_b5 import create_model as create_b5, load_and_preprocess_data as load_data_b5

def setup_hyperparameters():
    """Define hyperparameter combinations for training"""
    return {
        'learning_rates': [1e-3],  # 3 values
        'batch_sizes': [16],               # 2 values
        'dropout_rates': [0.2]            # 2 values
    }

def train_and_evaluate():
    """Main training and evaluation pipeline"""
    try:
        # Initialize tracking dictionary with correct structure
        tracking_dict = {
            'learning_rates': [],
            'batch_sizes': [],
            'dropout_rates': [],
            'val_accuracies': [],
            'val_losses': [],
            'training_times': [],
            'epochs_trained': [],
            'precision': [],
            'recall': [],
            'auc': [],
            'model_names': []
        }

        # Add epochs variable at the beginning of the function
        epochs = 10  # You can adjust this number
        
        # Define clinical features
        clinical_features = [
            'Adequacy',
            'Squamocolumnar junction visibility',
            'Transformation zone',
            'Original squamous epithelium',
            'Columnar epithelium',
            'Metaplastic squamous epithelium',
            'Location of the lesion',
            'Grade 1',
            'Grade 2',
            'Suspicious for invasion',
            'Aceto uptake',
            'Margins',
            'Vessels',
            'Lesion size',
            'Iodine uptake',
            'SwedeFinal'
        ]
        
        # Initialize components
        preprocessor = ColposcopyPreprocessor()
        lesion_analyzer = LesionAnalyzer()
        clinical_metrics = ClinicalMetrics()
        clinical_validator = ClinicalValidator()
        safety_monitor = SafetyMonitor()
        
        # Create experiment folder
        experiment_folder = create_experiment_folder(SAVE_FOLDER)
        print(f"Created experiment folder: {experiment_folder}")
        
        # Setup visualization folder
        viz_folder = setup_visualization_folder(experiment_folder)
        clinical_report = ClinicalReport(viz_folder)
        
        # Continue with existing pipeline...
        previous_model, previous_metrics = load_previous_best_model(viz_folder)
        
        # Training setup and execution
        hyperparameters = setup_hyperparameters()
        callbacks = setup_callbacks(experiment_folder)
        
        # Model versions to compare
        models = {
            'B0': (create_b0, load_data_b0),
            'B3': (create_b3, load_data_b3),
            'B4': (create_b4, load_data_b4),
        }
        
        all_histories = {}
        
        for model_name, (model_creator, data_loader) in models.items():
            print(f"\nTraining {model_name}")
            start_time = time.time()  # Add training time tracking
            
            # Load data with model-specific size
            processed_images, aligned_labels, aligned_metadata = data_loader()
            
            # Visualize metadata for this model
            visualize_metadata(aligned_metadata, viz_folder)
            
            # Create model-specific preprocessing
            meta_X_train = preprocess_metadata(aligned_metadata)
            
            # Create fresh splits
            train_data, val_data, test_data = prepare_data_splits(
                processed_images, aligned_labels, meta_X_train
            )
            
            # Unpack the data splits
            (X_train, meta_X_train, y_train) = train_data
            (X_val, meta_X_val, y_val) = val_data
            
            # Compute class weights for this specific split
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train.ravel()),  # Flatten the array to get unique classes
                y=y_train.ravel()  # Flatten the array for class weight computation
            )
            class_weights = dict(enumerate(class_weights))
            
            # Create and train model
            model = model_creator(meta_X_train.shape[1])
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss=WeightedBCE(neg_weight=2.0, pos_weight=1.0),
                metrics=['accuracy', specificity, tf.keras.metrics.Precision(name='precision')]
            )
            
            # Use balanced generator
            train_dataset = create_class_balanced_generator(X_train, meta_X_train, y_train)
            
            # Train with class weights
            batch_size = hyperparameters['batch_sizes'][0]
            
            # Prepare validation data with correct shape
            val_images = X_val
            val_metadata = meta_X_val
            # Keep validation labels as 1D array (shape: (None,))
            val_labels = y_val  # No reshaping needed
            
            history = model.fit(
                train_dataset,
                validation_data=((val_images, val_metadata), val_labels),
                epochs=epochs,
                steps_per_epoch=len(X_train)//batch_size,
                class_weight=class_weights,
                callbacks=callbacks
            )
            
            # Update tracking dictionary with proper values
            training_time = time.time() - start_time
            tracking_dict['model_names'].append(model_name)
            tracking_dict['learning_rates'].append(hyperparameters['learning_rates'][0])
            tracking_dict['batch_sizes'].append(batch_size)
            tracking_dict['dropout_rates'].append(hyperparameters['dropout_rates'][0])
            tracking_dict['val_accuracies'].append(max(history.history['val_accuracy']))
            tracking_dict['val_losses'].append(min(history.history['val_loss']))
            tracking_dict['training_times'].append(training_time)
            tracking_dict['epochs_trained'].append(len(history.history['val_accuracy']))
            tracking_dict['precision'].append(max(history.history.get('val_precision', [0])))
            tracking_dict['recall'].append(max(history.history.get('val_recall', [0])))
            tracking_dict['auc'].append(max(history.history.get('val_auc', [0])))

            # Save model and history
            model.save(os.path.join(viz_folder, f'{model_name}_model.h5'))
            all_histories[model_name] = history.history
            
            # Generate performance reports
            (X_test, meta_X_test, y_test) = test_data
            test_dataset = tf.data.Dataset.from_tensor_slices(
                ((X_test, meta_X_test), y_test)
            ).batch(batch_size)
            
            model_output_folder = os.path.join(viz_folder, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            
            generate_model_report(model, history, test_dataset, model_output_folder, model_name)
            
            # After data loading
            print(f"Image data shape: {X_train.shape}")
            print(f"Metadata shape: {meta_X_train.shape}")
            print(f"Labels shape: {y_train.shape}")
            
            # Should output:
            # Image data shape: (N, 224, 224, 3)
            # Metadata shape: (N, clinical_features_dim)
            # Labels shape: (N,)
        
        # Create comparison visualization
        create_model_comparison(all_histories, viz_folder)

        # After the search, visualize results
        print("\nGenerating hyperparameter analysis...")
        visualize_hyperparameter_effects(tracking_dict, viz_folder)

        # Save hyperparameter search results
        search_results = {
            'best_parameters': best_params,
            'best_accuracy': float(best_val_accuracy),
            'parameter_search': {
                'learning_rates_tested': hyperparameters['learning_rates'],
                'batch_sizes_tested': hyperparameters['batch_sizes'],
                'dropout_rates_tested': hyperparameters['dropout_rates'],
                'total_combinations': total_combinations,
                'tracking_metrics': tracking_dict
            }
        }

        with open(os.path.join(viz_folder, 'hyperparameter_search_results.json'), 'w') as f:
            json.dump(search_results, f, indent=4)

        # Evaluate best model
        print("\nEvaluating best model...")
        test_metrics = evaluate_model(best_model, test_data)
        
        # Compare with previous model if it exists
        if previous_model is not None:
            print("\nComparing with previous best model...")
            prev_test_metrics = evaluate_model(previous_model, test_data)
            improvement = {
                metric: test_metrics[metric] - prev_test_metrics[metric]
                for metric in test_metrics.keys()
            }
            print("\nImprovement over previous model:")
            for metric, value in improvement.items():
                print(f"{metric}: {value:+.4f}")
        
        # Save final results
        final_results = {
            'test_metrics': test_metrics,
            'best_hyperparameters': best_params,
            'training_history': {
                'accuracy': best_history.history['accuracy'],
                'val_accuracy': best_history.history['val_accuracy'],
                'loss': best_history.history['loss'],
                'val_loss': best_history.history['val_loss']
            }
        }
        
        save_json(final_results, os.path.join(experiment_folder, 'final_results.json'))
        
        # Save the best model with .keras extension
        model_save_path = os.path.join(experiment_folder, 'best_model.keras')
        best_model.save(model_save_path)
        print(f"Saved best model to: {model_save_path}")
        
        # Generate predictions
        y_pred_prob, y_pred = make_predictions(best_model, X_test, meta_X_test)
        
        # Save and visualize results
        save_detailed_metrics(y_test, y_pred, y_pred_prob, experiment_folder)
        visualize_results(best_history, viz_folder)
        visualize_model_performance(y_test, y_pred, y_pred_prob, viz_folder)
        visualize_learning_dynamics(best_history, viz_folder)

        # Add metadata visualizations
        visualize_metadata_distributions(aligned_metadata, viz_folder)
        
        # Add weight evolution visualization
        weight_changes = safe_compare_model_weights(previous_model, best_model, viz_folder)
        if weight_changes:
            visualize_weight_updates(weight_changes, viz_folder)
        
        # Add augmentation visualization
        sample_image = X_train[0]
        visualize_data_augmentation(sample_image, viz_folder)

        # Add clinical analysis
        visualize_clinical_patterns(aligned_metadata, viz_folder)
        
        # Analyze feature importance
        importance_dict = analyze_feature_importance(
            best_model, X_test, meta_X_test, clinical_features, viz_folder
        )
        
        # Model interpretation
        sample_idx = np.random.randint(len(X_test))
        visualize_model_interpretation(
            best_model,
            X_test[sample_idx],
            meta_X_test[sample_idx],
            clinical_features,
            viz_folder
        )

        # Generate final report
        performance_report = generate_performance_report(tracking_dict, viz_folder)
        save_json(performance_report, os.path.join(experiment_folder, 'performance_report.json'))

        # Statistical Analysis
        print("\nPerforming statistical analysis...")
        statistical_results = perform_statistical_tests(aligned_metadata, clinical_features)
        save_statistical_analysis(statistical_results, viz_folder)
        
        # Model Interpretation
        print("\nGenerating model interpretations...")
        interpreter = ModelInterpreter(best_model)
        
        # Analyze sample cases
        n_samples = 5
        sample_indices = np.random.choice(len(X_test), n_samples)
        for idx in sample_indices:
            # Generate GradCAM
            image = np.expand_dims(X_test[idx], 0)
            heatmap = interpreter.generate_gradcam(image)
            
            # Feature importance
            importance = interpreter.analyze_feature_importance(
                image, 
                np.expand_dims(meta_X_test[idx], 0),
                clinical_features
            )
            
            # Uncertainty estimation
            uncertainty = interpreter.generate_uncertainty(
                image,
                np.expand_dims(meta_X_test[idx], 0)
            )
            
            # Save interpretations
            interpretation_results = {
                'sample_id': int(idx),
                'feature_importance': importance,
                'uncertainty': uncertainty
            }
            save_json(interpretation_results, 
                     os.path.join(viz_folder, f'interpretation_sample_{idx}.json'))
        
        # Patient Analysis
        print("\nGenerating patient profiles...")
        analyzer = PatientAnalyzer(
            best_model,
            meta_X_test,
            clinical_features
        )
        
        # Generate profiles for test cases
        for idx in range(len(X_test)):
            profile = analyzer.generate_patient_profile(
                idx,
                meta_X_test,
                y_pred_prob
            )
            save_json(profile, 
                     os.path.join(viz_folder, f'patient_profile_{idx}.json'))

        # Visualize advanced analyses
        print("\nGenerating advanced visualizations...")
        visualize_statistical_results(statistical_results, viz_folder)
        visualize_model_interpretations(viz_folder, viz_folder)
        visualize_patient_profiles(viz_folder, viz_folder)
        visualize_combined_analysis(viz_folder)

        # After training, add clinical validation and safety monitoring
        print("\nPerforming clinical validation...")
        
        # Make predictions with uncertainty estimates
        y_pred_prob, y_pred = make_predictions(best_model, X_test, meta_X_test)
        uncertainty_results = interpreter.generate_uncertainty(X_test, meta_X_test)
        uncertainties = np.array([uncertainty_results['uncertainty']] if isinstance(uncertainty_results['uncertainty'], float) 
                               else uncertainty_results['uncertainty'])
        
        # Clinical metrics calculation
        clinical_results = clinical_metrics.calculate_clinical_metrics(
            y_test, y_pred_prob
        )
        
        # Clinical validation
        validation_results = clinical_validator.validate_model_safety(
            y_test, y_pred_prob, clinical_features=meta_X_test
        )
        
        # Safety monitoring
        safety_alerts, batch_stats = safety_monitor.monitor_batch_predictions(
            y_pred_prob, uncertainties, clinical_features=meta_X_test
        )
        
        # Generate clinical report
        print("\nGenerating clinical report...")
        prediction_results = {
            'prediction': float(np.mean(y_pred_prob)),
            'uncertainties': uncertainties.tolist() if hasattr(uncertainties, 'tolist') else [float(uncertainties)],
            'batch_statistics': batch_stats
        }
        
        report_path = clinical_report.generate_clinical_report(
            prediction_results,
            clinical_results,
            validation_results,
            patient_data=aligned_metadata
        )
        
        # Save all results
        results = {
            'clinical_metrics': clinical_results,
            'validation_results': validation_results,
            'safety_alerts': safety_alerts,
            'batch_statistics': batch_stats
        }
        
        save_json(results, os.path.join(experiment_folder, 'clinical_results.json'))
        
        print("\nTraining and validation completed successfully!")
        print(f"Results saved in: {experiment_folder}")
        print(f"Clinical report generated: {report_path}")
        
        # Create JSON structure visualization
        print("\nGenerating JSON structure visualization...")
        create_json_structure_visualization(experiment_folder, viz_folder)
        
        # If there are critical safety alerts, print them
        critical_alerts = [a for a in safety_alerts if any(
            alert['level'] == 'CRITICAL' for alert in a.get('alerts', [])
        )]
        if critical_alerts:
            print("\nWARNING: Critical safety alerts detected!")
            for alert in critical_alerts:
                print(f"Sample {alert['index']}: {alert['alerts']}")
        
        return best_model, best_history, test_metrics, results

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def generate_clinical_report(predictions, uncertainties, validation_results, safety_alerts, viz_folder):
    """Generate comprehensive clinical report"""
    try:
        # Convert inputs to lists if they're single values
        if isinstance(predictions, (float, int)):
            predictions = [predictions]
        if isinstance(uncertainties, (float, int)):
            uncertainties = [uncertainties]
            
        # Ensure all inputs are lists
        predictions = np.asarray(predictions).flatten().tolist()
        uncertainties = np.asarray(uncertainties).flatten().tolist()
        
        report_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'predictions': predictions,
            'uncertainties': uncertainties,
            'validation_results': validation_results,
            'safety_alerts': safety_alerts
        }
        
        # Save report
        report_path = os.path.join(viz_folder, 'clinical_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
        print("Clinical report generated successfully")
        return True
        
    except Exception as e:
        print(f"Error generating clinical report: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting transfer learning pipeline...")
    best_model, history, metrics, results = train_and_evaluate()
    
    if best_model is not None:
        print("\nFinal Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}") 