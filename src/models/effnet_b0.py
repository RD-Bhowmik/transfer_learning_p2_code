from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

MODEL_NAME = "EfficientNetB0"
INPUT_SIZE = 224

def create_model(clinical_features_dim):
    """Create B0-specific model"""
    img_input = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='image_input')
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=img_input)
    base_model.trainable = False  # Freeze base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    clinical_input = layers.Input(shape=(clinical_features_dim,), name='clinical_input')
    y = layers.Dense(256, activation='swish')(clinical_input)
    y = layers.BatchNormalization()(y)
    
    combined = layers.Concatenate()([x, y])
    outputs = layers.Dense(1, activation='sigmoid')(combined)
    
    return Model(inputs=[img_input, clinical_input], outputs=outputs)

def load_and_preprocess_data():
    """B0-specific data loading"""
    from src.data_processing import load_and_preprocess_data
    return load_and_preprocess_data(target_size=INPUT_SIZE) 