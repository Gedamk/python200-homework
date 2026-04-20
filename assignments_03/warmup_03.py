import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

os.makedirs("assignments_03/outputs", exist_ok=True)

# =========================================================
# Setup block required by assignment
# =========================================================
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("=== IRIS DATASET SETUP ===")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# --- Preprocessing ---
# =========================================================

# Q1
print("\n=== PREPROCESSING Q1 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2
print("\n=== PREPROCESSING Q2 ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Column means of X_train_scaled:")
print(X_train_scaled.mean(axis=0))

# We fit the scaler on X_train only so that information from the test set does not leak into preprocessing.

# =========================================================
# --- KNN ---
# =========================================================

# Q1
print("\n=== KNN Q1: Unscaled Data ===")
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)

knn_unscaled_accuracy = accuracy_score(y_test, y_pred_knn_unscaled)
print("Accuracy:", knn_unscaled_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred_knn_unscaled, target_names=iris.target_names))

# Q2
print("\n=== KNN Q2: Scaled Data ===")
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

knn_scaled_accuracy = accuracy_score(y_test, y_pred_knn_scaled)
print("Accuracy:", knn_scaled_accuracy)

# In this run, scaling slightly hurt performance (0.9333 vs 1.0000 unscaled).
# That can happen on Iris because the dataset is already small, clean, and well-separated, so scaling is not always beneficial on a single split.

# Q3
print("\n=== KNN Q3: 5-Fold CV on Unscaled Training Data ===")
cv_scores = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)

print("Fold scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Std CV score:", cv_scores.std())

# This result is more trustworthy than a single train/test split because it averages across multiple folds instead of relying on just one split.

# Q4
print("\n=== KNN Q4: Compare k values with 5-Fold CV ===")
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -1

for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k}, mean CV score={mean_score:.4f}")
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"Chosen k: {best_k} with mean CV score {best_score:.4f}")

# I would choose k=5 because it achieved the top mean CV score in this run and was selected first among the tied best values.

# =========================================================
# --- Classifier Evaluation ---
# =========================================================

# Q1
print("\n=== CLASSIFIER EVALUATION Q1 ===")
cm = confusion_matrix(y_test, y_pred_knn_unscaled)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix (Unscaled Iris)")
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png", bbox_inches="tight")
plt.close()

print("Confusion matrix:")
print(cm)

# In this run, the model did not confuse any pair of species because the confusion matrix is perfectly diagonal.

# =========================================================
# --- The sklearn API: Decision Trees ---
# =========================================================

# Q1
print("\n=== DECISION TREE Q1 ===")
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)
print("Accuracy:", tree_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# The Decision Tree accuracy (0.9667) is slightly lower than the unscaled KNN accuracy (1.0000) in this run.
# Because Decision Trees do not use distance calculations, scaled vs. unscaled data would usually make little or no difference.

# =========================================================
# --- Logistic Regression and Regularization ---
# =========================================================

print("\n=== LOGISTIC REGRESSION Q1 ===")
for c_value in [0.01, 1.0, 100]:
    # The assignment asks for solver='liblinear'. On current sklearn versions,
    # liblinear does not directly support multiclass Iris, so OneVsRestClassifier
    # preserves the intended one-vs-rest behavior while keeping liblinear.
    base_model = LogisticRegression(C=c_value, max_iter=1000, solver="liblinear")
    model = OneVsRestClassifier(base_model)
    model.fit(X_train_scaled, y_train)

    coef_total = sum(np.abs(est.coef_).sum() for est in model.estimators_)
    print(f"C={c_value}, total coefficient magnitude={coef_total:.6f}")

# As C increases, regularization becomes weaker, so the total coefficient magnitude should increase.
# This shows that regularization shrinks coefficients and helps limit model complexity.

# =========================================================
# --- PCA ---
# =========================================================

digits = load_digits()
X_digits = digits.data      # 1797 images, flattened to 64 features
y_digits = digits.target    # labels 0-9
images = digits.images      # same data as 8x8 images

# Q1
print("\n=== PCA Q1 ===")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.suptitle("Sample Digits 0-9")
plt.savefig("assignments_03/outputs/sample_digits.png", bbox_inches="tight")
plt.close()

# Q2
print("\n=== PCA Q2 ===")
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits")
plt.savefig("assignments_03/outputs/pca_2d_projection.png", bbox_inches="tight")
plt.close()

# Yes, same-digit images generally tend to cluster together in this 2D PCA space, although there is still some overlap.

# Q3
print("\n=== PCA Q3 ===")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("assignments_03/outputs/pca_variance_explained.png", bbox_inches="tight")
plt.close()

n_80 = np.argmax(cumulative_variance >= 0.80) + 1
print("Approximate number of components for 80% variance:", n_80)

# Approximately this many components are needed to explain 80% of the variance.

# Q4
print("\n=== PCA Q4 ===")

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

sample_indices = [0, 1, 2, 3, 4]
n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, len(sample_indices), figsize=(10, 10))

# Original row
for col, idx in enumerate(sample_indices):
    axes[0, col].imshow(images[idx], cmap="gray_r")
    axes[0, col].set_title(f"Orig {y_digits[idx]}")
    axes[0, col].axis("off")

# Reconstruction rows
for row, n_comp in enumerate(n_values, start=1):
    for col, idx in enumerate(sample_indices):
        recon = reconstruct_digit(idx, scores, pca, n_comp)
        axes[row, col].imshow(recon, cmap="gray_r")
        axes[row, col].set_title(f"n={n_comp}")
        axes[row, col].axis("off")

plt.suptitle("PCA Reconstructions")
plt.savefig("assignments_03/outputs/pca_reconstructions.png", bbox_inches="tight")
plt.close()

# The digits become clearly recognizable by a moderate number of components such as around 15,
# and that generally matches the point where the variance curve starts to level off.

print("\nWarmup complete. Check assignments_03/outputs for saved images.")