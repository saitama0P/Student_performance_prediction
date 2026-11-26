"""
=============================================================
MINI PROJECT: Student Performance Prediction
Using Linear Regression and PCA (Principal Component Analysis)
=============================================================

Objective:
To predict student academic performance (final grade) based on
various factors such as study time, attendance, previous scores,
and other academic/personal attributes using Linear Regression,
with PCA applied for dimensionality reduction.

Author: Sanchit Mallick
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set style for plots
sns.set_style("whitegrid")
np.random.seed(42)

print("="*60)
print("STUDENT PERFORMANCE PREDICTION")
print("Using Linear Regression and PCA")
print("="*60)

# -------------------------------------------------------------
# STEP 1: Generate/Load Dataset
# -------------------------------------------------------------
# For this project we create a synthetic dataset that mimics
# real student performance data. In practice you can replace
# this with a real dataset (e.g., UCI Student Performance Dataset)

print("\n[STEP 1] Generating Student Dataset...")

n_students = 500

data = {
    'study_hours_per_week': np.random.normal(15, 5, n_students).clip(0, 40),
    'attendance_percentage': np.random.normal(80, 12, n_students).clip(40, 100),
    'previous_grade': np.random.normal(65, 15, n_students).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_students).clip(3, 10),
    'extracurricular_hours': np.random.normal(5, 3, n_students).clip(0, 20),
    'parental_support': np.random.randint(1, 6, n_students),  # scale 1-5
    'internet_usage_hours': np.random.normal(4, 2, n_students).clip(0, 12),
    'tutoring_hours': np.random.normal(2, 2, n_students).clip(0, 10),
    'stress_level': np.random.randint(1, 11, n_students),  # scale 1-10
    'num_absences': np.random.poisson(5, n_students),
}

df = pd.DataFrame(data)

# Create target variable (final_grade) based on a realistic formula
# with some noise added to simulate real-world unpredictability
df['final_grade'] = (
    0.35 * df['study_hours_per_week'] +
    0.30 * df['attendance_percentage'] +
    0.25 * df['previous_grade'] +
    2.0 * df['parental_support'] +
    1.5 * df['tutoring_hours'] -
    1.2 * df['stress_level'] -
    0.8 * df['num_absences'] -
    0.3 * df['internet_usage_hours'] +
    np.random.normal(0, 5, n_students)  # random noise
)

# Normalize final grade to a realistic 0-100 scale
df['final_grade'] = ((df['final_grade'] - df['final_grade'].min()) /
                      (df['final_grade'].max() - df['final_grade'].min()) * 100)

print(f"Dataset created with {n_students} student records")
print(f"Features: {list(df.columns[:-1])}")
print(f"Target: final_grade")

print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

# -------------------------------------------------------------
# STEP 2: Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
print("\n[STEP 2] Performing Exploratory Data Analysis...")

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Student Performance Factors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.close()
print("✓ Correlation heatmap saved")

# Distribution of final grades
plt.figure(figsize=(8, 5))
sns.histplot(df['final_grade'], kde=True, color='steelblue', bins=30)
plt.title('Distribution of Student Final Grades', fontsize=14, fontweight='bold')
plt.xlabel('Final Grade')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig('grade_distribution.png', dpi=150)
plt.close()
print("✓ Grade distribution plot saved")

# -------------------------------------------------------------
# STEP 3: Data Preprocessing
# -------------------------------------------------------------
print("\n[STEP 3] Preprocessing Data...")

X = df.drop('final_grade', axis=1)
y = df['final_grade']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} students")
print(f"Testing set size: {X_test.shape[0]} students")

# Standardize features (important before applying PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features standardized (mean=0, std=1)")

# -------------------------------------------------------------
# STEP 4: Baseline Linear Regression (WITHOUT PCA)
# -------------------------------------------------------------
print("\n[STEP 4] Training Baseline Linear Regression (without PCA)...")

lr_baseline = LinearRegression()
lr_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = lr_baseline.predict(X_test_scaled)

mse_baseline = mean_squared_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mse_baseline)
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

print(f"Baseline Model Performance:")
print(f"  R² Score: {r2_baseline:.4f}")
print(f"  RMSE: {rmse_baseline:.4f}")
print(f"  MAE: {mae_baseline:.4f}")

# -------------------------------------------------------------
# STEP 5: Apply PCA for Dimensionality Reduction
# -------------------------------------------------------------
print("\n[STEP 5] Applying PCA for Dimensionality Reduction...")

# First, determine optimal number of components using explained variance
pca_full = PCA()
pca_full.fit(X_train_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nExplained Variance by Each Principal Component:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")

# Plot explained variance (Scree Plot)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio,
        color='teal', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance,
         marker='o', color='darkorange')
plt.axhline(y=0.95, color='red', linestyle='--', label='95% Variance Threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=150)
plt.close()
print("✓ PCA variance plots saved")

# Select number of components that explain 90% variance
# (capped so PCA actually demonstrates dimensionality reduction)
n_components = min(np.argmax(cumulative_variance >= 0.90) + 1, X_train_scaled.shape[1] - 2)
print(f"\nSelected number of components (capturing ~90% variance, reduced for demonstration): {n_components}")

# Apply PCA with selected components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✓ Dimensionality reduced from {X_train_scaled.shape[1]} to {n_components} features")

# -------------------------------------------------------------
# STEP 6: Linear Regression WITH PCA
# -------------------------------------------------------------
print("\n[STEP 6] Training Linear Regression with PCA-transformed data...")

lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)

mse_pca = mean_squared_error(y_test, y_pred_pca)
rmse_pca = np.sqrt(mse_pca)
mae_pca = mean_absolute_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)

print(f"PCA Model Performance:")
print(f"  R² Score: {r2_pca:.4f}")
print(f"  RMSE: {rmse_pca:.4f}")
print(f"  MAE: {mae_pca:.4f}")

# -------------------------------------------------------------
# STEP 7: Comparison of Both Models
# -------------------------------------------------------------
print("\n[STEP 7] Model Comparison")
print("="*60)
print(f"{'Metric':<20}{'Without PCA':<20}{'With PCA':<20}")
print("-"*60)
print(f"{'R² Score':<20}{r2_baseline:<20.4f}{r2_pca:<20.4f}")
print(f"{'RMSE':<20}{rmse_baseline:<20.4f}{rmse_pca:<20.4f}")
print(f"{'MAE':<20}{mae_baseline:<20.4f}{mae_pca:<20.4f}")
print(f"{'Num Features':<20}{X_train_scaled.shape[1]:<20}{n_components:<20}")
print("="*60)

# Visualization: Actual vs Predicted (both models)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_baseline, alpha=0.6, color='steelblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2)
axes[0].set_xlabel('Actual Grade')
axes[0].set_ylabel('Predicted Grade')
axes[0].set_title(f'Without PCA (R² = {r2_baseline:.3f})')

axes[1].scatter(y_test, y_pred_pca, alpha=0.6, color='darkorange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2)
axes[1].set_xlabel('Actual Grade')
axes[1].set_ylabel('Predicted Grade')
axes[1].set_title(f'With PCA (R² = {r2_pca:.3f})')

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.close()
print("\n✓ Actual vs Predicted comparison plot saved")

# -------------------------------------------------------------
# STEP 8: Feature Importance (from baseline model coefficients)
# -------------------------------------------------------------
print("\n[STEP 8] Feature Importance Analysis (Baseline Model)")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_baseline.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
colors_bar = ['green' if c > 0 else 'red' for c in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors_bar)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Predicting Student Performance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("✓ Feature importance plot saved")

# -------------------------------------------------------------
# STEP 9: Predict Performance for a New Student (Demo)
# -------------------------------------------------------------
print("\n[STEP 9] Demo Prediction for a New Student")

new_student = pd.DataFrame({
    'study_hours_per_week': [20],
    'attendance_percentage': [90],
    'previous_grade': [75],
    'sleep_hours': [7],
    'extracurricular_hours': [4],
    'parental_support': [4],
    'internet_usage_hours': [3],
    'tutoring_hours': [3],
    'stress_level': [4],
    'num_absences': [2]
})

new_student_scaled = scaler.transform(new_student)
predicted_grade_baseline = lr_baseline.predict(new_student_scaled)[0]

new_student_pca = pca.transform(new_student_scaled)
predicted_grade_pca = lr_pca.predict(new_student_pca)[0]

print("New Student Profile:")
print(new_student.to_string(index=False))
print(f"\nPredicted Final Grade (Without PCA): {predicted_grade_baseline:.2f}")
print(f"Predicted Final Grade (With PCA): {predicted_grade_pca:.2f}")

print("\n" + "="*60)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("="*60)
#done by Sanchit,Kratika,Farhan,Siddhi 🤙