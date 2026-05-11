from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURE_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total",
]

COLUMNS = FEATURE_NAMES + ["spam_label"]

def load_spambase():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    return pd.read_csv(url, header=None, names=COLUMNS)

def save_boxplot(df, feature, filename):
    plt.figure(figsize=(7, 5))
    data = [
        df.loc[df["spam_label"] == 0, feature],
        df.loc[df["spam_label"] == 1, feature],
    ]
    plt.boxplot(data, tick_labels=["Ham", "Spam"], showfliers=False)
    plt.title(f"{feature}: Ham vs Spam")
    plt.xlabel("Class")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

print("\n--- Task 1: Load and Explore ---")

df = load_spambase()

print("Dataset shape:", df.shape)
print("Number of emails:", len(df))
print("\nClass counts:")
print(df["spam_label"].value_counts())
print("\nClass percentages:")
print(df["spam_label"].value_counts(normalize=True).round(3))

# Comment:
# The classes are not perfectly balanced.
# Accuracy is useful, but precision and recall are also important for spam detection.

for feature, filename in [
    ("word_freq_free", "boxplot_word_freq_free.png"),
    ("char_freq_!", "boxplot_char_freq_exclamation.png"),
    ("capital_run_length_total", "boxplot_capital_run_length_total.png"),
]:
    save_boxplot(df, feature, filename)
    print("Saved:", filename)

print("\nFeature scale summary:")
print(df[FEATURE_NAMES].describe().loc[["mean", "std", "min", "max"]])

# Comment:
# Many emails have zero for many word-frequency features.
# Feature scales vary a lot, so scaling matters for KNN, logistic regression, and PCA.

print("\n--- Task 2: Prepare Your Data ---")

X = df[FEATURE_NAMES]
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comment:
# The scaler is fit only on training data to avoid leaking information from the test set.

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(0.90, linestyle="--")
plt.axvline(n_components_90, linestyle="--")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Spambase PCA Variance Explained")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "spambase_pca_variance_explained.png")
plt.close()

print("Number of components for 90% variance:", n_components_90)
print("Saved: spambase_pca_variance_explained.png")

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

# Comment:
# PCA must happen after scaling because PCA is affected by feature scale.

print("\n--- Task 3: Classifier Comparison ---")

results = {}

def evaluate_model(name, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "y_pred": y_pred,
    }

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    return model, y_pred, acc

evaluate_model("KNN unscaled", KNeighborsClassifier(n_neighbors=5), X_train, X_test)
evaluate_model("KNN scaled", KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled)
evaluate_model("KNN PCA", KNeighborsClassifier(n_neighbors=5), X_train_pca, X_test_pca)

print("\nDecision Tree depth comparison")

for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)
    print(f"max_depth={depth}, train accuracy={train_acc:.4f}, test accuracy={test_acc:.4f}")

# Comment:
# As depth increases, training accuracy increases.
# If training accuracy is much higher than test accuracy, the model is overfitting.

chosen_tree_depth = 10

tree_model, tree_pred, tree_acc = evaluate_model(
    "Decision Tree depth 10",
    DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=RANDOM_STATE),
    X_train,
    X_test,
)

rf_model, rf_pred, rf_acc = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    X_train,
    X_test,
)

evaluate_model(
    "Logistic Regression scaled",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled,
    X_test_scaled,
)

evaluate_model(
    "Logistic Regression PCA",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca,
    X_test_pca,
)

# Comment:
# The best model should be chosen by looking at accuracy, precision, recall, F1-score, and the confusion matrix.
# For spam filters, false positives are important because a real email marked as spam can be harmful.

best_name = max(results, key=lambda name: results[name]["accuracy"])
best_result = results[best_name]

print("\nBest model by accuracy:", best_name)
print("Best accuracy:", best_result["accuracy"])

cm = confusion_matrix(y_test, best_result["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot()
plt.title(f"Best Model Confusion Matrix: {best_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "best_model_confusion_matrix.png")
plt.close()

print("Saved: best_model_confusion_matrix.png")
print(cm)

tree_importances = pd.Series(tree_model.feature_importances_, index=FEATURE_NAMES)
rf_importances = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES)

print("\nTop 10 Decision Tree features:")
print(tree_importances.sort_values(ascending=False).head(10))

print("\nTop 10 Random Forest features:")
print(rf_importances.sort_values(ascending=False).head(10))

top_rf = rf_importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_rf.sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importances.png")
plt.close()

print("Saved: feature_importances.png")

# Comment:
# Important spam features often include words like free or money, and punctuation like ! or $.

print("\n--- Task 4: Cross-Validation ---")

cv_models = {
    "KNN unscaled": (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN scaled": (KNeighborsClassifier(n_neighbors=5), X_train_scaled),
    "KNN PCA": (KNeighborsClassifier(n_neighbors=5), X_train_pca),
    "Decision Tree depth 10": (
        DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=RANDOM_STATE),
        X_train,
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        X_train,
    ),
    "Logistic Regression scaled": (
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_scaled,
    ),
    "Logistic Regression PCA": (
        LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
        X_train_pca,
    ),
}

for name, (model, X_cv) in cv_models.items():
    scores = cross_val_score(model, X_cv, y_train, cv=5)
    print(f"{name:30s} mean={scores.mean():.4f}, std={scores.std():.4f}")

# Comment:
# Cross-validation is more reliable than a single split.
# The most stable model has the lowest standard deviation across folds.

print("\n--- Task 5: Building a Prediction Pipeline ---")

best_tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
])

best_non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
])

best_tree_pipeline.fit(X_train, y_train)
tree_pipe_pred = best_tree_pipeline.predict(X_test)

print("\nBest tree-based pipeline: Random Forest")
print(classification_report(y_test, tree_pipe_pred))

best_non_tree_pipeline.fit(X_train, y_train)
non_tree_pipe_pred = best_non_tree_pipeline.predict(X_test)

print("\nBest non-tree pipeline: Scaled Logistic Regression")
print(classification_report(y_test, non_tree_pipe_pred))

pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
])

pca_pipeline.fit(X_train, y_train)
pca_pipe_pred = pca_pipeline.predict(X_test)

print("\nPCA Logistic Regression pipeline")
print(classification_report(y_test, pca_pipe_pred))

# Comment:
# Tree pipelines do not need scaling because trees split by feature thresholds.
# Non-tree pipelines often need scaling because distance and coefficients are affected by feature size.
# Pipelines package preprocessing and modeling together, making the work easier to reuse and deploy.

print("\nProject complete.")
