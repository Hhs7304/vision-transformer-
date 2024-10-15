import tensorflow as tf
from model import build_vit_model
from dataset_preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
(X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

# Build and compile the model
vit_model = build_vit_model(image_size=32, patch_size=4, num_classes=10)
vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = vit_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
vit_model.evaluate(X_test, y_test)

# Predict on test data
y_pred = vit_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("Classification Report: \n", classification_report(y_test, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
