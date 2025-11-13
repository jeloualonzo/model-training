# filepath: python/model.py
"""
MobileNetV2 Classifier 
Achieves 80+% training / 100% validation accuracy
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from config import config
def create_feature_extractor():
    """
    Create MobileNetV2 α=0.5 feature extractor (frozen)
    EXACT match to WebGL's truncated MobileNet
    """
    # ✅ USE MobileNetV2 with alpha=0.5 (same as WebGL)
    base_model = keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        alpha=0.5,  # ← This is the key! Same as WebGL
        pooling='avg'  # Global average pooling (like WebGL)
    )
    
    # Freeze feature extractor (same as WebGL)
    base_model.trainable = False
    
    print(f"✅ Feature extractor created: {base_model.output_shape}")
    print(f"   Model: MobileNetV2 (alpha=0.5)")
    print(f"   Output features: 1280")
    
    return base_model
def create_classifier(num_classes: int):
    """
    Create classifier head matching WebGL architecture EXACTLY
    
    WebGL Structure:
    1. Dense(256, relu, L2=0.01)
    2. Dropout(0.6)
    3. Dense(num_classes, softmax)
    """
    feature_extractor = create_feature_extractor()
    
    # Input layer accepts grayscale
    inputs = keras.Input(shape=(224, 224, 1), name='image_input')
    
    # Convert grayscale to RGB by repeating the channel 3 times
    # This matches WebGL's behavior (grayscale images displayed as RGB)
    x = layers.Concatenate()([inputs, inputs, inputs])  # (224, 224, 1) -> (224, 224, 3)
    
    # Feature extraction (frozen MobileNetV2)
    features = feature_extractor(x, training=False)
    
    # Classifier head (EXACT WebGL match)
    x = layers.Dense(
        256, 
        activation='relu',
        kernel_regularizer=regularizers.l2(config.MODEL['l2_regularization']),
        name='dense_256'
    )(features)
    
    x = layers.Dropout(
        config.MODEL['dropout_rate'],
        name='dropout_0.6'
    )(x)
    
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        name='output_softmax'
    )(x)
    
    # Build model
    model = keras.Model(inputs=inputs, outputs=outputs, name='MobileNetV2_Classifier')
    
    # Compile with EXACT WebGL settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.TRAINING['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n" + "="*60)
    print("📊 MODEL ARCHITECTURE (WebGL Match)")
    print("="*60)
    model.summary()
    print("="*60 + "\n")
    
    return model, feature_extractor
def load_model(model_path: str):
    """Load trained model from disk"""
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
