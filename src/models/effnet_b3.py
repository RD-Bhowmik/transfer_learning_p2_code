from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model

MODEL_NAME = "EfficientNetB3"
INPUT_SIZE = 300

def create_model(clinical_features_dim):
    """Create B3-specific model"""
    # Image pathway
    img_input = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3), name='image_input')
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=img_input)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Clinical pathway
    clinical_input = layers.Input(shape=(clinical_features_dim,), name='clinical_input')
    y = layers.Dense(256, activation='swish')(clinical_input)
    y = layers.BatchNormalization()(y)
    
    # Feature fusion
    combined = layers.Concatenate()([x, y])
    outputs = layers.Dense(1, activation='sigmoid')(combined)
    
    return Model(inputs=[img_input, clinical_input], outputs=outputs)

def load_and_preprocess_data():
    """B3-specific data loading"""
    from src.data_processing import load_and_preprocess_data
    return load_and_preprocess_data(target_size=INPUT_SIZE) 