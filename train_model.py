import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle
from imblearn.over_sampling import SMOTE

# Set up correct file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'healthcare-dataset-stroke-data.csv')
model_save_path = os.path.join(current_dir, 'stroke_model.pkl')

print(f"Loading dataset from {dataset_path}")

# Load the dataset
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Display basic info
print("\nDataset Information:")
print(f"Number of stroke cases: {df['stroke'].sum()} ({df['stroke'].mean() * 100:.2f}%)")
print(f"Missing values:\n{df.isnull().sum()}")

# Drop id column if it exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define preprocessing for numeric features
numeric_features = ['age', 'avg_glucose_level', 'bmi']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                        'work_type', 'Residence_type', 'smoking_status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Handle class imbalance with SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
print(f"Training set after SMOTE: {X_train_resampled.shape[0]} samples")
print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

# Create and train the model
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, 
                               max_depth=10,
                               min_samples_split=5,
                               min_samples_leaf=2,
                               class_weight='balanced',
                               random_state=42)

model.fit(X_train_resampled, y_train_resampled)

# Create complete pipeline for saving
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test_preprocessed)
y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', model)
])
cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model on all data
print("\nTraining final model on all data...")
# Fit preprocessor on all data
X_all_preprocessed = preprocessor.fit_transform(X)
# Apply SMOTE
X_all_resampled, y_all_resampled = smote.fit_resample(X_all_preprocessed, y)
# Fit model
model.fit(X_all_resampled, y_all_resampled)

# Create the final pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Save the model
print(f"\nSaving model to {model_save_path}")
try:
    with open(model_save_path, 'wb') as file:
        pickle.dump(final_pipeline, file)
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}") 