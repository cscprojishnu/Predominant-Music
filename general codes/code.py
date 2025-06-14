import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Paths to data directories
train_dir = r"D:\DJT\Downloads\PAPER\PAPER 3\trainingset"
test_dir = r"D:\DJT\Downloads\PAPER\PAPER 3\testset"

# Image dimensions and batch size
IMG_HEIGHT = 48  # Updated based on IMAGE_SIZE
IMG_WIDTH = 48  # Updated based on IMAGE_SIZE
BATCH_SIZE = 128  # Updated based on BATCH_SIZE

# Data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Prepare training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Prepare testing dataset
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model with updated architecture
DIMENSIONS = 256  # Based on DIMENSIONS
model = models.Sequential([
    layers.Conv2D(DIMENSIONS, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(DIMENSIONS, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(DIMENSIONS, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(DIMENSIONS, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model with updated optimizer settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=WEIGHT_DECAY)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with updated epochs
EPOCHS = 20  # Updated from the new hyperparameters
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# Plot accuracy vs. epochs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss vs. epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Loss Value: {loss * 100:.2f}%")

# Generate classification report
test_generator.reset()
y_pred = np.argmax(model.predict(test_generator), axis=-1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

'''# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(11, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()'''

# Save the model
model.save(r"D:\DJT\Downloads\PAPER\PAPER 3\processed_data\data_new.h5")
