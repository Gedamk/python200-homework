import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

os.makedirs("assignments_02/outputs", exist_ok=True)

# --- scikit-learn API ---
# Q1
print("=== SCIKIT-LEARN Q1 ===")
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict(np.array([[4]]))[0]
pred_8 = model.predict(np.array([[8]]))[0]

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", pred_4)
print("Predicted salary for 8 years:", pred_8)

# Q2
print("\n=== SCIKIT-LEARN Q2 ===")
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)
print("Reshaped shape:", x_2d.shape)

# scikit-learn needs X to be 2D because each row is a sample and each column is a feature.

# Q3
print("\n=== SCIKIT-LEARN Q3 ===")
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print("Cluster centers:")
print(kmeans.cluster_centers_)
print("Points per cluster:")
print(np.bincount(labels))

plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200
)
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("assignments_02/outputs/kmeans_clusters.png", bbox_inches="tight")
plt.close()

# --- Linear Regression ---
np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1
print("\n=== LINEAR REGRESSION Q1 ===")
plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("assignments_02/outputs/cost_vs_age.png", bbox_inches="tight")
plt.close()

# There appear to be two distinct groups, suggesting smoker status has a strong effect on cost.

# Q2
print("\n=== LINEAR REGRESSION Q2 ===")
X_age = age.reshape(-1, 1)
y_cost = cost

X_train, X_test, y_train, y_test = train_test_split(
    X_age, y_cost, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
print("\n=== LINEAR REGRESSION Q3 ===")
model_age = LinearRegression()
model_age.fit(X_train, y_train)

y_pred = model_age.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model_age.score(X_test, y_test)

print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)
print("RMSE:", rmse)
print("R^2 on test set:", r2)

# The slope means each additional year of age is associated with an increase in predicted medical cost.

# Q4
print("\n=== LINEAR REGRESSION Q4 ===")
X_full = np.column_stack([age, smoker])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("R^2 using age only:", r2)
print("R^2 using age + smoker:", r2_full)
print("age coefficient:   ", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])

# The smoker coefficient represents the average extra cost for smokers compared with non-smokers, holding age fixed.

# Q5
print("\n=== LINEAR REGRESSION Q5 ===")
y_pred_full = model_full.predict(X_test_full)

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_full, y_test_full)
plt.plot(
    [y_test_full.min(), y_test_full.max()],
    [y_test_full.min(), y_test_full.max()],
    color="red"
)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# A point above the diagonal means the true value is higher than the prediction.
# A point below the diagonal means the prediction is higher than the true value.

print("\nWarmup 02 complete.")