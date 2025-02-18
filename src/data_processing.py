import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from collections import Counter
import traceback  # Add this import at the top
import tensorflow as tf
import albumentations as A
import cv2
import math
from sklearn.utils.class_weight import compute_class_weight

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our config
from src.config import POSITIVE_FOLDER, NEGATIVE_FOLDER, METADATA_PATH
from src.data_loader import load_images_from_folder, load_metadata, display_dataset_info

def extract_case_id_from_filename(filename):
    """Extract CaseID from filename"""
    try:
        # Convert filename to string and get basename
        filename = str(filename)
        basename = os.path.basename(filename)
        
        # Remove extension
        name_without_ext = os.path.splitext(basename)[0]
        
        # Split by underscore if present
        parts = name_without_ext.split('_')
        
        # Get the part that contains letters (the CaseID)
        case_id = None
        for part in parts:
            if any(c.isalpha() for c in part):
                # Extract only alphabetic characters
                case_id = ''.join(c for c in part if c.isalpha())
                break
        
        # If no case_id found in parts, try the whole name
        if not case_id:
            case_id = ''.join(c for c in name_without_ext if c.isalpha())
        
        # Convert to uppercase for consistency
        case_id = case_id.upper() if case_id else None
        
        if not case_id:
            print(f"Warning: Could not extract CaseID from {filename}")
            return None
            
        return case_id
        
    except Exception as e:
        print(f"Error processing filename {filename}: {str(e)}")
        return None

def align_data(images, labels, filenames, metadata, target_size):
    """Align images with metadata including image size"""
    try:
        # Add image size to metadata
        metadata = metadata.copy()
        metadata['image_size'] = target_size
        
        # Convert metadata CaseID to uppercase and ensure string type
        metadata["CaseID"] = metadata["CaseID"].astype(str).str.strip().str.upper()
        
        # Extract CaseIDs from image filenames
        image_case_ids = []
        valid_indices = []
        
        for idx, fname in enumerate(filenames):
            case_id = extract_case_id_from_filename(fname)
            if case_id:
                image_case_ids.append(case_id)
                valid_indices.append(idx)
        
        # Create metadata lookup dictionary
        metadata_dict = metadata.set_index('CaseID').to_dict('index')
        
        # Initialize aligned arrays
        aligned_images = []
        aligned_labels = []
        aligned_metadata_rows = []
        
        # Align data
        for idx, case_id in zip(valid_indices, image_case_ids):
            if case_id in metadata_dict:
                aligned_images.append(images[idx])
                aligned_labels.append(labels[idx])
                aligned_metadata_rows.append(metadata_dict[case_id])
        
        # Convert to numpy arrays/DataFrame
        aligned_images = np.array(aligned_images)
        aligned_labels = np.array(aligned_labels)
        aligned_metadata = pd.DataFrame(aligned_metadata_rows)
        
        print(f"\nAligned data shapes:")
        print(f"Images: {aligned_images.shape}")
        print(f"Labels: {aligned_labels.shape}")
        print(f"Metadata: {aligned_metadata.shape}")
        
        return aligned_images, aligned_labels, aligned_metadata
        
    except Exception as e:
        print(f"Error in align_data: {str(e)}")
        traceback.print_exc()
        raise

def preprocess_metadata(metadata):
    """Preprocess metadata with expanded feature handling"""
    try:
        # Define features with proper categorical handling
        categorical_features = [
            'Adequacy', 'Lesion size', 'Iodine uptake',
            'Squamocolumnar junction visibility', 'Transformation zone',
            'Aceto uptake', 'Margins', 'Vessels'
        ]
        
        numerical_features = ['SwedeFinal', 'image_size']
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
            
        return preprocessor.fit_transform(metadata)
    
    except Exception as e:
        print(f"Error in preprocess_metadata: {str(e)}")
        raise

def setup_data_augmentation():
    """Setup data augmentation for training"""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

def load_and_preprocess_data(target_size=224, verbose=True):
    """Load and preprocess all data with size control"""
    try:
        if verbose:
            print("Starting data loading process...")
        
        # Load metadata first to fail fast if there are issues
        metadata = load_metadata(METADATA_PATH)
        if metadata is None or len(metadata) == 0:
            raise ValueError("Failed to load valid metadata")
        
        # Load images with labels
        positive_images, positive_labels, positive_filenames = load_images_from_folder(POSITIVE_FOLDER, label=1, target_size=target_size)
        negative_images, negative_labels, negative_filenames = load_images_from_folder(NEGATIVE_FOLDER, label=0, target_size=target_size)
        
        # Combine datasets
        all_images = np.concatenate([positive_images, negative_images])
        all_labels = np.concatenate([positive_labels, negative_labels])
        all_filenames = positive_filenames + negative_filenames
        
        # Display dataset information
        display_dataset_info(all_images, all_labels, metadata)
        
        # Verify data alignment
        print("\nVerifying data alignment:")
        print(f"Number of images: {len(all_images)}")
        print(f"Number of labels: {len(all_labels)}")
        print(f"Number of filenames: {len(all_filenames)}")
        print(f"Number of metadata entries: {len(metadata)}")
        
        if not (len(all_images) == len(all_labels) == len(all_filenames)):
            raise ValueError("Mismatch in data lengths")
            
        # Debug print first few filenames
        print("\nSample filenames:")
        for i, fname in enumerate(all_filenames[:5]):
            print(f"{i}: {fname}")
            
        # Align data with metadata
        print("\nAligning data with metadata...")
        aligned_images, aligned_labels, aligned_metadata = align_data(all_images, all_labels, all_filenames, metadata, target_size)
        
        if verbose:
            print("\nDataset Information:")
            print(f"Number of images: {len(all_images)}")
        
        return aligned_images, aligned_labels, aligned_metadata
        
    except Exception as e:
        print(f"\nError in load_and_preprocess_data: {str(e)}")
        raise

def prepare_data_splits(all_images_aligned, all_labels_aligned, metadata_features_processed, test_size=0.3, val_size=0.5):
    """Prepare train, validation, and test splits"""
    try:
        # Convert metadata_features_processed to numpy array if it's not already
        if not isinstance(metadata_features_processed, np.ndarray):
            metadata_features_processed = np.array(metadata_features_processed)
        
        # Ensure all inputs are numpy arrays with correct shapes
        X = np.array(all_images_aligned)
        y = np.array(all_labels_aligned)
        y = y.reshape(-1) if len(y.shape) > 1 else y  # Ensure labels are 1D
        meta_X = metadata_features_processed

        print(f"\nData shapes before splitting:")
        print(f"Images: {X.shape}")
        print(f"Labels: {y.shape}")
        print(f"Metadata: {meta_X.shape}")

        # First split: train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        meta_X_train, meta_X_temp = train_test_split(
            meta_X, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        
        # Second split: validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_temp
        )
        
        meta_X_val, meta_X_test = train_test_split(
            meta_X_temp, 
            test_size=val_size, 
            random_state=42, 
            stratify=y_temp
        )
        
        # Print split information
        print(f"\nDataset Splits:")
        print(f"Training set: {X_train.shape}, {meta_X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {meta_X_val.shape}, {y_val.shape}")
        print(f"Test set: {X_test.shape}, {meta_X_test.shape}, {y_test.shape}")
        
        print("\nClass Distribution:")
        print("Training set:", Counter(y_train))
        print("Validation set:", Counter(y_val))
        print("Test set:", Counter(y_test))
        
        return (X_train, meta_X_train, y_train), (X_val, meta_X_val, y_val), (X_test, meta_X_test, y_test)
        
    except Exception as e:
        print(f"Error in prepare_data_splits: {str(e)}")
        print(f"Shapes: images={np.array(all_images_aligned).shape}, "
              f"labels={np.array(all_labels_aligned).shape}, "
              f"metadata={metadata_features_processed.shape}")
        raise

def create_medical_augmentation_pipeline():
    """Create an augmentation pipeline specific for colposcopy images"""
    return A.Compose([
        # Color adjustments (preserve medical features)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,  # Minimal hue change to preserve diagnostic colors
                p=0.5
            ),
        ], p=0.7),
        
        # Medical-specific transformations
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=0.5),  # Enhance local contrast
            A.Sharpen(alpha=(0.2, 0.5), p=0.5),  # Enhance edges
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Simulate focus variation
        ], p=0.5),
        
        # Lesion-preserving spatial transformations
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-15, 15),
                p=0.5
            ),
        ], p=0.7),
        
        # Quality simulation
        A.OneOf([
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.5),
        ], p=0.3),
    ])

def validate_image_quality(image):
    """Validate image quality before processing"""
    try:
        quality_metrics = {
            'resolution': {
                'width': image.shape[1],
                'height': image.shape[0],
                'adequate': min(image.shape[:2]) >= 224
            },
            'contrast': {
                'value': measure_image_contrast(image),
                'adequate': measure_image_contrast(image) > 0.4
            },
            'noise_level': {
                'value': estimate_noise_level(image),
                'acceptable': estimate_noise_level(image) < 0.1
            },
            'blur_detection': {
                'value': detect_blur_level(image),
                'is_sharp': detect_blur_level(image) < 100
            }
        }
        
        return quality_metrics
        
    except Exception as e:
        print(f"Error in image quality validation: {str(e)}")
        return None

def measure_image_contrast(image):
    """Measure image contrast"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(np.std(image.astype(float)))

def estimate_noise_level(image):
    """Estimate image noise level"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    H, W = image.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(image, -1, np.array(M)))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    
    return float(sigma)

def detect_blur_level(image):
    """Detect image blur level using Laplacian variance"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return cv2.Laplacian(image, cv2.CV_64F).var()

def augment_medical_image(image, augmentation):
    """Apply medical-specific augmentation to an image"""
    # Ensure image is in correct format
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Apply augmentation
    augmented = augmentation(image=image)
    
    # Return normalized image
    return augmented['image'].astype(np.float32) / 255.0

def create_augmentation_dataset(images, metadata, labels, batch_size=32):
    """Create a dataset with medical-specific augmentation"""
    augmentation = create_medical_augmentation_pipeline()
    
    def augment_batch(images, metadata, labels):
        # Augment images
        augmented_images = tf.py_function(
            lambda x: np.stack([augment_medical_image(img, augmentation) for img in x]),
            [images],
            tf.float32
        )
        augmented_images.set_shape([None, 224, 224, 3])
        
        # Ensure labels have correct shape
        labels = tf.reshape(labels, (-1, 1))
        
        # Return both image and metadata inputs
        return (augmented_images, metadata), labels
    
    # Create dataset with both images and metadata
    dataset = tf.data.Dataset.from_tensor_slices((images, metadata, labels))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def load_images_from_folder(folder, label, target_size):
    """Load images with configurable size"""
    images = []
    valid_files = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = load_img(img_path, target_size=(target_size, target_size))
            images.append(img_to_array(img))
            valid_files.append(filename)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
    return np.array(images), np.full(len(images), label), valid_files

def create_class_balanced_generator(image_data, metadata, labels, batch_size=32):
    """Create balanced data generator"""
    # Ensure labels are flattened for class weight computation
    flat_labels = labels.ravel() if isinstance(labels, np.ndarray) else labels
    class_weights = compute_class_weight('balanced', classes=np.unique(flat_labels), y=flat_labels)
    sample_weights = np.array([class_weights[label] for label in flat_labels])
    
    # Create balanced dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_data, metadata, labels))
    # Keep labels as 1D array (shape: (batch_size,))
    dataset = dataset.map(lambda img, meta, lbl: ((img, meta), lbl))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    
    return dataset

if __name__ == "__main__":
    print("Testing data processing module...")
    print(f"Positive folder path: {POSITIVE_FOLDER}")
    print(f"Negative folder path: {NEGATIVE_FOLDER}")
    print(f"Metadata path: {METADATA_PATH}")
    
    # Test data loading
    try:
        all_images, all_labels, all_filenames, metadata = load_and_preprocess_data()
        print("Data loading successful!")
    except Exception as e:
        print(f"Error loading data: {str(e)}") 