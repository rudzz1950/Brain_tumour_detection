#!/usr/bin/env python
# Enhanced Brain Tumor Detection Pipeline with Training and Visualization

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.utils import Sequence

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ======================================
# 1. Data Loading and Preprocessing
# ======================================

class BrainTumorDataset(Sequence):
    """Data generator for brain tumor segmentation"""
    def __init__(self, json_file, root_dir, batch_size=4, image_size=(256, 256), augment=False):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        
        # Data augmentation
        self.augmenter = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        with open(json_file) as f:
            self.data = json.load(f)
            
        self.image_keys = list(self.data.keys())
        
    def __len__(self):
        return len(self.image_keys) // self.batch_size
    
    def __getitem__(self, idx):
        batch_keys = self.image_keys[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []
        
        for key in batch_keys:
            image_info = self.data[key]
            file_path = os.path.join(self.root_dir, image_info['filename'])
            
            # Load and preprocess image
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size) / 255.0
            
            # Create mask
            mask = np.zeros(self.image_size, dtype=np.uint8)
            for region in image_info['regions']:
                shape_attrs = region['shape_attributes']
                if shape_attrs['name'] == 'polygon':
                    points_x = shape_attrs['all_points_x']
                    points_y = shape_attrs['all_points_y']
                    points = np.array(list(zip(points_x, points_y)), dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)
            
            mask = cv2.resize(mask, self.image_size)
            mask = np.expand_dims(mask, axis=-1)
            
            # Apply data augmentation if enabled
            if self.augment and np.random.random() > 0.5:
                augmented = self.augmenter.get_random_transform(img_shape=image.shape)
                image = self.augmenter.apply_transform(image, augmented)
                mask = self.augmenter.apply_transform(mask[..., 0][..., np.newaxis], augmented)
            
            images.append(image)
            masks.append(mask)
            
        return np.array(images), np.array(masks)

class ClassificationDataset(Sequence):
    """Data generator for tumor classification"""
    def __init__(self, data_dir, batch_size=32, image_size=(150, 150), augment=False, split='train', test_size=0.2):
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        
        # Data augmentation
        self.augmenter = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Load and split data
        self.classes = ['no', 'yes']  # no tumor, tumor
        self.images = []
        self.labels = []
        
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, split, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label_idx)
        
        # Split into train/val if needed
        if split == 'train':
            self.images, _, self.labels, _ = train_test_split(
                self.images, self.labels, test_size=test_size, stratify=self.labels, random_state=42
            )
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        for img_path in batch_x:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size) / 255.0
            
            if self.augment and np.random.random() > 0.5:
                img = self.augmenter.random_transform(img)
                
            images.append(img)
            
        return np.array(images), tf.keras.utils.to_categorical(batch_y, num_classes=len(self.classes))

# ======================================
# 2. Model Definitions
# ======================================

def build_segmentation_model(input_shape=(256, 256, 3)):
    """Build the ResUNet model for tumor segmentation"""
    # [Previous ResUNet implementation...]
    # ... (include the ResUNet, ASPP, and AttentionBlock classes from previous code)
    
    inputs = layers.Input(shape=input_shape)
    model = ResUNet()(inputs)
    return models.Model(inputs, model, name='segmentation_model')

def build_classification_model(input_shape=(150, 150, 3), num_classes=2):
    """Build the EfficientNetB0 based classification model"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs, name='classification_model')

# ======================================
# 3. Training Functions
# ======================================

def train_segmentation_model(train_json, val_json, data_dir, epochs=50, batch_size=4):
    """Train the segmentation model"""
    # Create data generators
    train_gen = BrainTumorDataset(train_json, data_dir, batch_size=batch_size, augment=True)
    val_gen = BrainTumorDataset(val_json, data_dir, batch_size=batch_size)
    
    # Build and compile model
    model = build_segmentation_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    
    # Callbacks
    callbacks = [
        callbacks.ModelCheckpoint(
            'best_segmentation_model.h5',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

def train_classification_model(data_dir, epochs=30, batch_size=32):
    """Train the classification model"""
    # Create data generators
    train_gen = ClassificationDataset(data_dir, batch_size=batch_size, augment=True, split='train')
    val_gen = ClassificationDataset(data_dir, batch_size=batch_size, split='val')
    
    # Build and compile model
    model = build_classification_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        callbacks.ModelCheckpoint(
            'best_classification_model.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return model, history

# ======================================
# 4. Visualization Utilities
# ======================================

def plot_training_history(history, model_name):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'{model_name} Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(images, masks, preds, num_samples=3):
    """Visualize model predictions"""
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(preds[i].squeeze() > 0.5, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ======================================
# 5. Main Pipeline
# ======================================

class BrainTumorPipeline:
    def __init__(self, seg_model_path=None, cls_model_path=None):
        self.seg_model = None
        self.cls_model = None
        
        if seg_model_path:
            self.seg_model = tf.keras.models.load_model(seg_model_path, compile=False)
        if cls_model_path:
            self.cls_model = tf.keras.models.load_model(cls_model_path, compile=False)
    
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path.copy()
            if len(image.shape) == 2:  # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image / 255.0
    
    def segment_tumor(self, image, threshold=0.5):
        """Segment tumor from brain MRI"""
        if self.seg_model is None:
            raise ValueError("Segmentation model not loaded")
            
        # Resize and preprocess for segmentation
        original_shape = image.shape[:2]
        seg_input = cv2.resize(image, (256, 256))
        seg_input = np.expand_dims(seg_input, axis=0)
        
        # Predict and resize back to original
        mask = self.seg_model.predict(seg_input)[0]
        mask = cv2.resize(mask.squeeze(), (original_shape[1], original_shape[0]))
        
        return (mask > threshold).astype(np.uint8)
    
    def classify_tumor(self, image, mask):
        """Classify tumor from segmented region"""
        if self.cls_model is None:
            raise ValueError("Classification model not loaded")
            
        # Apply mask and prepare for classification
        masked_image = image * np.repeat(mask[..., np.newaxis], 3, axis=-1)
        cls_input = cv2.resize(masked_image, (150, 150))
        cls_input = np.expand_dims(cls_input, axis=0)
        
        # Predict and get class probabilities
        pred = self.cls_model.predict(cls_input)[0]
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        
        return {
            'class': 'Tumor' if class_idx == 1 else 'No Tumor',
            'confidence': confidence,
            'probabilities': pred.tolist()
        }
    
    def process_image(self, image_path, visualize=True):
        """Process an image through the full pipeline"""
        try:
            # Load and preprocess image
            image = self.preprocess_image(image_path)
            original_image = image.copy()
            
            # Segment tumor
            if self.seg_model:
                mask = self.segment_tumor(image)
                
                # Classify tumor if segmentation is successful
                classification = None
                if self.cls_model and np.any(mask > 0):
                    classification = self.classify_tumor(image, mask)
                
                # Visualize results
                if visualize:
                    self.visualize_results(original_image, mask, classification)
                
                return {
                    'image': original_image,
                    'mask': mask,
                    'classification': classification
                }
            else:
                raise ValueError("Segmentation model not loaded")
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
    
    def visualize_results(self, image, mask, classification=None):
        """Visualize segmentation and classification results"""
        plt.figure(figsize=(15, 5))
        
        # Original image with mask overlay
        plt.subplot(131)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.3, cmap='jet')
        plt.title('Original Image with Tumor Mask')
        plt.axis('off')
        
        # Segmentation mask
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Tumor Segmentation')
        plt.axis('off')
        
        # Classification results
        plt.subplot(133)
        if classification:
            classes = ['No Tumor', 'Tumor']
            probs = classification['probabilities']
            
            plt.bar(classes, probs, color=['red', 'green'])
            plt.title(f'Classification: {classification["class"]}\nConfidence: {classification["confidence"]:.2f}')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No classification available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# ======================================
# 6. Example Usage
# ======================================

def train_models():
    """Example function to train both models"""
    print("Training segmentation model...")
    seg_model, seg_history = train_segmentation_model(
        train_json='path/to/train_annotations.json',
        val_json='path/to/val_annotations.json',
        data_dir='path/to/images',
        epochs=50,
        batch_size=4
    )
    plot_training_history(seg_history, 'Segmentation Model')
    
    print("\nTraining classification model...")
    cls_model, cls_history = train_classification_model(
        data_dir='path/to/classification_data',
        epochs=30,
        batch_size=32
    )
    plot_training_history(cls_history, 'Classification Model')

def run_inference():
    """Example function to run inference on a single image"""
    # Initialize pipeline with trained models
    pipeline = BrainTumorPipeline(
        seg_model_path='best_segmentation_model.h5',
        cls_model_path='best_classification_model.h5'
    )
    
    # Process an image
    result = pipeline.process_image('path/to/brain_scan.jpg')
    
    if result:
        print(f"Tumor detected: {result['classification']['class']}")
        print(f"Confidence: {result['classification']['confidence']:.2f}")

if __name__ == "__main__":
    # Uncomment to train models
    # train_models()
    
    # Run inference on example image
    run_inference()
