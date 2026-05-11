from pathlib import Path

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
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

print("\n--- Preprocessing ---")

# Q1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nPreprocessing Q1")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q2
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing Q2")
print("Mean of scaled X_train columns:")
print(np.round(X_train_scaled.mean(axis=0), 6))

# Comment:
# I fit the scaler on X_train only to avoid leaking test data information into training.

print("\n--- KNN ---")

# Q1
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN Q1: Unscaled")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_unscaled))
print(classification_report(y_test, y_pred_knn_unscaled, target_names=iris.target_names))

# Q2
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN Q2: Scaled")
print("Accuracy:", accuracy_score(y_test, y_pred_knn_scaled))

# Comment:
# Scaling may make little difference for Iris because the feature ranges are not extremely different.
# KNN uses distance, so scaling can matter more when feature scales are very different.

# Q3
cv_scores = cross_val_score(knn_unscaled, X_train, y_train, cv=5)

print("\nKNN Q3: Cross-validation")
print("Fold scores:", cv_scores)
print("Mean:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Comment:
# Cross-validation is more trustworthy than one split because it tests the model on multiple splits.

# Q4
print("\nKNN Q4: Testing different k values")

best_k = None
best_score = -1

for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"k={k}, mean CV accuracy={mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"Chosen k: {best_k}")

# Comment:
# I would choose the k value with the best mean cross-validation score.

print("\n--- Classifier Evaluation ---")

# Q1
cm = confusion_matrix(y_test, y_pred_knn_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "knn_confusion_matrix.png")
plt.close()

print("\nSaved outputs/knn_confusion_matrix.png")
print(cm)

# Comment:
# If the model confuses any species, it is usually versicolor and virginica.
# Setosa is usually easier to separate.

print("\n--- Decision Trees ---")

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("\nDecision Tree Q1")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# Comment:
# I compare Decision Tree to KNN by looking at their accuracy and reports.
# Decision Trees do not depend on distance, so scaling usually does not affect them.

print("\n--- Logistic Regression and Regularization ---")

for C in [0.01, 1.0, 100]:
    base_model = LogisticRegression(C=C, max_iter=1000, solver="liblinear")
    model = OneVsRestClassifier(base_model)
    model.fit(X_train_scaled, y_train)

    coef_size = sum(np.abs(estimator.coef_).sum() for estimator in model.estimators_)
    print(f"C={C}, total coefficient magnitude={coef_size:.4f}")

# Comment:
# The assignment asks for solver='liblinear'. In this sklearn version, liblinear needs
# one-vs-rest wrapping for multiclass Iris classification.
# As C increases, regularization becomes weaker and the coefficient magnitude usually increases.
# Regularization helps keep model weights smaller and can reduce overfitting.

print("\n--- PCA ---")

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Q1
print("\nPCA Q1")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(12, 2))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_digits.png")
plt.close()

print("Saved outputs/sample_digits.png")

# Q2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits PCA 2D Projection")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_2d_projection.png")
plt.close()

print("Saved outputs/pca_2d_projection.png")

# Comment:
# Some same-digit images cluster together, but there is overlap because handwriting varies.

# Q3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components_80 = np.argmax(cumulative_variance >= 0.80) + 1

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(0.80, linestyle="--")
plt.axvline(components_80, linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Variance Explained")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_variance_explained.png")
plt.close()

print("Saved outputs/pca_variance_explained.png")
print("Components needed for 80% variance:", components_80)

# Comment:
# The printed number shows approximately how many components explain 80% of the variance.

# Q4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]
sample_indices = list(range(5))

fig, axes = plt.subplots(len(n_values) + 1, len(sample_indices), figsize=(10, 10))

for col, sample_idx in enumerate(sample_indices):
    axes[0, col].imshow(images[sample_idx], cmap="gray_r")
    axes[0, col].set_title("Original")
    axes[0, col].axis("off")

for row, n in enumerate(n_values, start=1):
    for col, sample_idx in enumerate(sample_indices):
        reconstruction = reconstruct_digit(sample_idx, scores, pca, n)
        axes[row, col].imshow(reconstruction, cmap="gray_r")
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_reconstructions.png")
plt.close()

print("Saved outputs/pca_reconstructions.png")

# Comment:
# The digits become clearly recognizable around 15 components.
# More components add more detail, which matches the variance curve leveling off.

print("\nWarmup complete.")
