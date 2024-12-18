import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
np.random.seed(42)

from tensorflow.keras import layers, models

base_path = r"C:\Users\20006\.cache\kagglehub\datasets\prasunroy\natural-images\versions\1\data\natural_images"

classes = sorted(os.listdir(base_path))
print(f"Classes: {classes}")


image_size = (128, 128)
data = []
labels = []

for class_index, class_name in enumerate(classes):
    class_path = os.path.join(base_path, class_name)
    for filename in os.listdir(class_path):
        file_path = os.path.join(class_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                img = img.resize(image_size)
                data.append(np.array(img))
                labels.append(class_index)

data = np.array(data)
labels = np.array(labels)
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")


labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])

num_classes = 8
one_hot_labels = np.eye(num_classes)[labels]

print(one_hot_labels)

print(f"One-hot encoded labels shape: {y_train.shape}")


model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # 8 output classes
])


X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
