# The CSV uses semicolons as separators, so pd.read_csv needs sep=';'.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.makedirs("assignments_02/outputs", exist_ok=True)

print("=== TASK 1: LOAD AND EXPLORE ===")
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

plt.figure(figsize=(8, 6))
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.savefig("assignments_02/outputs/g3_distribution.png", bbox_inches="tight")
plt.close()

print("\n=== TASK 2: PREPROCESS THE DATA ===")
print("Original shape:", df.shape)

df_clean = df[df["G3"] != 0].copy()
print("Filtered shape:", df_clean.shape)
print("Rows removed:", len(df) - len(df_clean))

# Keeping G3=0 rows would distort the model because those zeros represent students who missed
# the final exam, not true academic performance.

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

df_original_encoded = df.copy()
for col in yes_no_cols:
    df_original_encoded[col] = df_original_encoded[col].map({"yes": 1, "no": 0})
df_original_encoded["sex"] = df_original_encoded["sex"].map({"F": 0, "M": 1})

corr_original = df_original_encoded["absences"].corr(df_original_encoded["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Absences/G3 correlation on original data:", corr_original)
print("Absences/G3 correlation on filtered data:", corr_filtered)

# Filtering changes the result because many G3=0 rows come from students who missed the final exam.
# Their absences and zero grades can distort the real relationship between absences and performance.

print("\n=== TASK 3: EXPLORATORY DATA ANALYSIS ===")
numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc", "schoolsup", "internet",
    "higher", "activities", "sex"
]

corrs = df_clean[numeric_cols + ["G3"]].corr(numeric_only=True)["G3"].drop("G3").sort_values()
print("Correlations with G3:")
print(corrs)

# Plot 1: failures vs G3
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("Failures vs G3")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/failures_vs_g3.png", bbox_inches="tight")
plt.close()

# This plot should show that students with more past failures tend to have lower final grades.

# Plot 2: studytime vs G3
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("Study Time vs G3")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png", bbox_inches="tight")
plt.close()

# This plot helps show whether more weekly study time is associated with higher final grades.

print("\n=== TASK 4: BASELINE MODEL ===")
X_base = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)
rmse_base = np.sqrt(np.mean((y_pred_base - y_test) ** 2))
r2_base = baseline_model.score(X_test, y_test)

print("Slope:", baseline_model.coef_[0])
print("RMSE:", rmse_base)
print("R^2:", r2_base)

# Since grades are on a 0-20 scale, RMSE shows the typical prediction error in grade points.
# The slope shows how much predicted G3 changes for each additional past failure.

print("\n=== TASK 5: FULL MODEL ===")
feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("Train R^2:", train_r2)
print("Test R^2:", test_r2)
print("RMSE:", rmse)
print("Baseline Test R^2:", r2_base)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# If train R^2 and test R^2 are close, the model generalizes reasonably well.
# A large gap would suggest overfitting.

print("\n=== TASK 6: EVALUATE AND SUMMARIZE ===")
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png", bbox_inches="tight")
plt.close()

# A point above the diagonal means the actual grade is higher than predicted.
# A point below the diagonal means the model predicted too high.
# If errors spread similarly across the range, error is roughly uniform.

print("Filtered dataset size:", df_clean.shape)
print("Test set size:", X_test.shape[0])

sorted_coefs = sorted(zip(feature_cols, model.coef_), key=lambda x: x[1])
print("Most negative coefficients:", sorted_coefs[:2])
print("Most positive coefficients:", sorted_coefs[-2:])

# Coefficient interpretation:
# The top two positive coefficients are internet and higher.
# In this model, internet access has the strongest positive coefficient, about +0.834.
# This means students with internet access are predicted to have higher final math grades,
# assuming the other features stay the same.
# The second strongest positive coefficient is higher, about +0.610.
# This means students who plan to pursue higher education are also predicted to have
# higher final math grades, assuming the other features stay the same.
#
# The bottom two negative coefficients are schoolsup and failures.
# The most negative coefficient is schoolsup, about -2.062.
# This means students receiving extra school support are predicted to have lower final
# math grades, assuming the other features stay the same. This does not necessarily mean
# school support causes lower grades; it may mean students receiving support were already struggling.
# The second most negative coefficient is failures, about -1.145.
# This means students with more past class failures are predicted to have lower final grades.

# Plain-language summary:
# After filtering out students with G3 equal to 0, the filtered dataset had 357 rows and 18 columns.
# The test set had 72 rows, meaning the model was evaluated on 72 student records that it did not
# see during training.
#
# The best full model had an RMSE of about 2.86.
# Since final grades are on a 0-20 scale, this means the model's typical prediction error is
# about 2.86 grade points. For example, if the model predicts a final grade of 12, the actual
# grade might commonly be around 9.14 to 14.86.
#
# The best full model had a test R^2 score of about 0.154.
# In plain English, this means the model explains about 15.4% of the variation in students'
# final math grades. This is better than the baseline model, but it also shows that many
# important factors are still missing from the model.
#
# One result that surprised me was that adding G1 increased the test R^2 to about 0.749.
# This surprised me because the model became much stronger after adding only one feature.
# It makes sense because G1 is the student's first-period grade, so it is already a strong
# signal of how the student may perform on the final grade G3.
print("\n=== NEGLECTED FEATURE: G1 ===")
feature_cols_g1 = feature_cols + ["G1"]
X_g1 = df_clean[feature_cols_g1].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

print("Test R^2 with G1 added:", model_g1.score(X_test_g1, y_test_g1))

# A high R^2 with G1 does not mean G1 causes G3.
# It is useful for later prediction, but not for early intervention before G1 exists.

print("\nProject 02 complete.")