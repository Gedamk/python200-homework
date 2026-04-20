import os
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
    ConfusionMatrixDisplay
)

os.makedirs("assignments_03/outputs", exist_ok=True)

print("=== TASK 1: LOAD AND EXPLORE ===")

columns = [
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
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam_label"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None, names=columns)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["spam_label"].value_counts())
print("\nClass proportions:")
print(df["spam_label"].value_counts(normalize=True))

# The classes are somewhat imbalanced, so raw accuracy alone can be misleading.
# A model could look good on accuracy while still making too many mistakes on the spam class.

features_to_plot = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    df.boxplot(column=feature, by="spam_label")
    plt.title(f"{feature} by Spam Label")
    plt.suptitle("")
    plt.xlabel("spam_label (0 = ham, 1 = spam)")
    plt.ylabel(feature)
    safe_name = feature.replace("!", "exclamation")
    plt.savefig(f"assignments_03/outputs/{safe_name}_boxplot.png", bbox_inches="tight")
    plt.close()

print("\nFeature summary:")
print(df[features_to_plot].describe())

# Many word-frequency features are zero for most emails, which means the data is sparse and heavily skewed.
# The numeric scales vary a lot because some features are percentages/frequencies, while others are counts
# or run-length statistics that can become very large. This matters for models like KNN and logistic regression
# because large-scale features can dominate distance calculations or coefficient estimation.

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

print("\n=== TASK 2: PREPARE DATA ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# We fit the scaler on training data only to avoid data leakage from the test set.

pca = PCA()
pca.fit(X_train_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.argmax(cum_var >= 0.90) + 1

print("Number of PCA components to reach 90% variance:", n_90)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cum_var) + 1), cum_var)
plt.axhline(0.90, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Spambase PCA Cumulative Explained Variance")
plt.grid(True)
plt.savefig("assignments_03/outputs/spambase_pca_variance.png", bbox_inches="tight")
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_90]

def evaluate_model(name, model, X_train_data, X_test_data, y_train_data, y_test_data):
    model.fit(X_train_data, y_train_data)
    y_pred = model.predict(X_test_data)
    acc = accuracy_score(y_test_data, y_pred)

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(y_test_data, y_pred))

    return model, y_pred, acc

print("\n=== TASK 3: CLASSIFIER COMPARISON ===")

knn_unscaled, knn_unscaled_pred, knn_unscaled_acc = evaluate_model(
    "KNN (Unscaled)",
    KNeighborsClassifier(n_neighbors=5),
    X_train, X_test, y_train, y_test
)

knn_scaled, knn_scaled_pred, knn_scaled_acc = evaluate_model(
    "KNN (Scaled)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, X_test_scaled, y_train, y_test
)

knn_pca, knn_pca_pred, knn_pca_acc = evaluate_model(
    "KNN (Scaled + PCA)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca, X_test_pca, y_train, y_test
)

print("\n=== Decision Tree Depth Comparison ===")
for depth in [3, 5, 10, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"max_depth={depth}, train_accuracy={train_acc:.4f}, test_accuracy={test_acc:.4f}")

# As depth increases, training accuracy usually rises, but test accuracy may stop improving or drop.
# That is a sign of overfitting: the tree memorizes training details instead of generalizing well.

chosen_depth = 5

decision_tree, tree_pred, tree_acc = evaluate_model(
    f"Decision Tree (max_depth={chosen_depth})",
    DecisionTreeClassifier(max_depth=chosen_depth, random_state=42),
    X_train, X_test, y_train, y_test
)

# I would choose depth=5 for production because it usually balances strong performance and lower overfitting
# better than a fully unbounded tree.

random_forest, rf_pred, rf_acc = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)

logreg_scaled, logreg_scaled_pred, logreg_scaled_acc = evaluate_model(
    "Logistic Regression (Scaled)",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_scaled, X_test_scaled, y_train, y_test
)

logreg_pca, logreg_pca_pred, logreg_pca_acc = evaluate_model(
    "Logistic Regression (Scaled + PCA)",
    LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"),
    X_train_pca, X_test_pca, y_train, y_test
)

print("\n=== Task 3 Summary Comment ===")
print("For spam filtering, accuracy is useful, but false positives and false negatives matter more.")
print("A false positive marks a real email as spam; a false negative lets spam into the inbox.")
print("For many users, minimizing false positives is especially important so important email is not lost.")

model_scores = {
    "KNN Unscaled": knn_unscaled_acc,
    "KNN Scaled": knn_scaled_acc,
    "KNN PCA": knn_pca_acc,
    "Decision Tree": tree_acc,
    "Random Forest": rf_acc,
    "LogReg Scaled": logreg_scaled_acc,
    "LogReg PCA": logreg_pca_acc,
}

best_model_name = max(model_scores, key=model_scores.get)
print("\nBest model by test accuracy:", best_model_name)

best_pred = None
if best_model_name == "KNN Unscaled":
    best_pred = knn_unscaled_pred
elif best_model_name == "KNN Scaled":
    best_pred = knn_scaled_pred
elif best_model_name == "KNN PCA":
    best_pred = knn_pca_pred
elif best_model_name == "Decision Tree":
    best_pred = tree_pred
elif best_model_name == "Random Forest":
    best_pred = rf_pred
elif best_model_name == "LogReg Scaled":
    best_pred = logreg_scaled_pred
else:
    best_pred = logreg_pca_pred

cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Best Model Confusion Matrix: {best_model_name}")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png", bbox_inches="tight")
plt.close()

print("\nBest model confusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"False positives: {fp}")
print(f"False negatives: {fn}")

print("\n=== Feature Importances ===")

tree_importances = pd.Series(
    decision_tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Features:")
print(tree_importances.head(10))

rf_importances = pd.Series(
    random_forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Features:")
print(rf_importances.head(10))

plt.figure(figsize=(10, 6))
rf_importances.head(10).sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Random Forest Feature Importances")
plt.savefig("assignments_03/outputs/feature_importances.png", bbox_inches="tight")
plt.close()

# The tree and random forest often agree on some important features,
# though the random forest is usually more stable because it averages many trees.

print("\n=== TASK 4: CROSS-VALIDATION ===")

cv_models = {
    "KNN Unscaled": (KNeighborsClassifier(n_neighbors=5), X_train),
    "KNN Scaled": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        X_train
    ),
    "KNN Scaled + PCA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_90)),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        X_train
    ),
    "Decision Tree": (DecisionTreeClassifier(max_depth=chosen_depth, random_state=42), X_train),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train),
    "LogReg Scaled": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
        ]),
        X_train
    ),
    "LogReg Scaled + PCA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_90)),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
        ]),
        X_train
    )
}

for name, (model, X_data) in cv_models.items():
    scores = cross_val_score(model, X_data, y_train, cv=5)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

print("\n=== TASK 5: PREDICTION PIPELINES ===")

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

tree_pipeline.fit(X_train, y_train)
tree_pipeline_pred = tree_pipeline.predict(X_test)

print("\nTree-based pipeline classification report:")
print(classification_report(y_test, tree_pipeline_pred))

non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

non_tree_pipeline.fit(X_train, y_train)
non_tree_pred = non_tree_pipeline.predict(X_test)

print("\nNon-tree-based pipeline classification report:")
print(classification_report(y_test, non_tree_pred))

# Tree and non-tree pipelines may not have the same structure because tree-based models
# usually do not need scaling or PCA, while models like logistic regression and KNN often do.
# Pipelines are valuable because they package preprocessing and modeling together, making the
# workflow safer, easier to reuse, and easier to hand off or deploy.

print("\nProject complete. Check assignments_03/outputs for generated figures.")