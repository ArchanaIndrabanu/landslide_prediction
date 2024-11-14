# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the dataset
file_path = '/content/MatrixTaiwan.csv'
df = pd.read_csv(file_path)

# Step 2: Data preprocessing
# Drop unnecessary columns
columns_to_drop = ['intercept', 'ID_SU', 'Lithology', 'NorthM', 'NorthStd',
                   'EastM', 'EastStd', 'Year', 'ExpandedSlide',
                   'ExpandedActSlide', 'ActiveSlide']
df.drop(columns=columns_to_drop, inplace=True)

# Handle missing values by filling with the mean of each column
df.fillna(df.mean(), inplace=True)

# Separate features (X) and binary target (y)
X = df.drop(columns=['Label'])
y = df['Label']

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert targets to binary (0 or 1)
y_label = np.where(y > 0, 1, 0)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_label)

# Split the balanced data into training and testing sets
X_train, X_test, y_train_label, y_test_label = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 3: Build a more complex neural network model
def build_complex_model(input_shape):
    model = Sequential()

    # First hidden layer
    model.add(Dense(256, input_dim=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Second hidden layer
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Third hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fourth hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build the model
complex_model = build_complex_model(X_train.shape[1])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Step 4: Train the model
history = complex_model.fit(
    X_train, y_train_label,
    epochs=300,
    batch_size=512,
    validation_split=0.1,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=2
)

# Step 5: Evaluate the model
# Calculate overall training accuracy
train_predictions = complex_model.predict(X_train)
train_predictions_binary = (train_predictions > 0.5).astype(int)
training_accuracy = accuracy_score(y_train_label, train_predictions_binary)
print(f"Overall Training Accuracy: {training_accuracy}")

# Predict labels on test data
y_pred_label = complex_model.predict(X_test)
y_pred_label_binary = (y_pred_label > 0.5).astype(int)

# Calculate accuracy for the classification model on the test set
test_accuracy = accuracy_score(y_test_label, y_pred_label_binary)
print(f"Overall Test Accuracy: {test_accuracy}")

# Print confusion matrix and classification report
cm = confusion_matrix(y_test_label, y_pred_label_binary)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test_label, y_pred_label_binary)
print("Classification Report:")
print(report)
