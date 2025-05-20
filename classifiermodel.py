from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Paths to your dataset folders
train_path = "D:/videodata/data/train"
val_path = "D:/videodata/data/val"

# Data generators with rescaling
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# Load train and validation data
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Define the model
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Predict on validation data (needed for MAE, MSE)
val_data.reset()  # important to reset before prediction
preds = model.predict(val_data)
preds_labels = (preds > 0.5).astype(int).flatten()

# True labels from the generator
true_labels = val_data.classes

# Calculate Accuracy manually (for confirmation)
accuracy = np.mean(preds_labels == true_labels)

# Calculate MAE and MSE between predicted probabilities and true labels
mae = mean_absolute_error(true_labels, preds.flatten())
mse = mean_squared_error(true_labels, preds.flatten())

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation MSE: {mse:.4f}")
