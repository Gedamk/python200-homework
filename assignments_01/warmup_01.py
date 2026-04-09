import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


print("\n" + "=" * 60)
print("PANDAS REVIEW")
print("=" * 60)

# --- Pandas Q1 ---
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}
df = pd.DataFrame(data)

print("\n--- Pandas Q1 ---")
print("First 3 rows:")
print(df.head(3))
print(f"Shape: {df.shape}")
print("Data types:")
print(df.dtypes)

# --- Pandas Q2 ---
print("\n--- Pandas Q2 ---")
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print(filtered_df)

# --- Pandas Q3 ---
print("\n--- Pandas Q3 ---")
df["grade_curved"] = df["grade"] + 5
print(df)

# --- Pandas Q4 ---
print("\n--- Pandas Q4 ---")
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

# --- Pandas Q5 ---
print("\n--- Pandas Q5 ---")
mean_by_city = df.groupby("city")["grade"].mean()
print(mean_by_city)

# --- Pandas Q6 ---
print("\n--- Pandas Q6 ---")
df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

# --- Pandas Q7 ---
print("\n--- Pandas Q7 ---")
sorted_df = df.sort_values(by="grade", ascending=False)
print(sorted_df.head(3))


print("\n" + "=" * 60)
print("NUMPY REVIEW")
print("=" * 60)

# --- NumPy Q1 ---
print("\n--- NumPy Q1 ---")
arr1 = np.array([10, 20, 30, 40, 50])
print("Array:", arr1)
print("Shape:", arr1.shape)
print("Dtype:", arr1.dtype)
print("Ndim:", arr1.ndim)

# --- NumPy Q2 ---
print("\n--- NumPy Q2 ---")
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
print(arr2)
print("Shape:", arr2.shape)
print("Size:", arr2.size)

# --- NumPy Q3 ---
print("\n--- NumPy Q3 ---")
print(arr2[:2, :2])

# --- NumPy Q4 ---
print("\n--- NumPy Q4 ---")
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))
print(zeros_arr)
print(ones_arr)

# --- NumPy Q5 ---
print("\n--- NumPy Q5 ---")
arr3 = np.arange(0, 50, 5)
print("Array:", arr3)
print("Shape:", arr3.shape)
print("Mean:", np.mean(arr3))
print("Sum:", np.sum(arr3))
print("Standard Deviation:", np.std(arr3))

# --- NumPy Q6 ---
print("\n--- NumPy Q6 ---")
random_normal = np.random.normal(0, 1, 200)
print("Mean:", np.mean(random_normal))
print("Std:", np.std(random_normal))


print("\n" + "=" * 60)
print("MATPLOTLIB REVIEW")
print("=" * 60)

# --- Matplotlib Q1 ---
print("\n--- Matplotlib Q1 ---")
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- Matplotlib Q2 ---
print("\n--- Matplotlib Q2 ---")
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]
plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.show()

# --- Matplotlib Q3 ---
print("\n--- Matplotlib Q3 ---")
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.figure()
plt.scatter(x1, y1, label="Dataset 1")
plt.scatter(x2, y2, label="Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Matplotlib Q4 ---
print("\n--- Matplotlib Q4 ---")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subject")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()


print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS REVIEW")
print("=" * 60)

# --- Descriptive Stats Q1 ---
print("\n--- Descriptive Stats Q1 ---")
data_stats = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print("Mean:", np.mean(data_stats))
print("Median:", np.median(data_stats))
print("Variance:", np.var(data_stats))
print("Standard Deviation:", np.std(data_stats))

# --- Descriptive Stats Q2 ---
print("\n--- Descriptive Stats Q2 ---")
scores_dist = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(scores_dist, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# --- Descriptive Stats Q3 ---
print("\n--- Descriptive Stats Q3 ---")
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# --- Descriptive Stats Q4 ---
print("\n--- Descriptive Stats Q4 ---")
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.show()

# The exponential distribution is more skewed.
# The mean works well for roughly normal data.
# The median is often better for skewed data.

# --- Descriptive Stats Q5 ---
print("\n--- Descriptive Stats Q5 ---")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mode1 = stats.mode(data1, keepdims=False)
mode2 = stats.mode(data2, keepdims=False)

print("Data1 Mean:", np.mean(data1))
print("Data1 Median:", np.median(data1))
print("Data1 Mode:", mode1.mode)

print("Data2 Mean:", np.mean(data2))
print("Data2 Median:", np.median(data2))
print("Data2 Mode:", mode2.mode)

# The mean is much larger for data2 because 150 is an outlier
# that pulls the average upward. The median is more resistant.


print("\n" + "=" * 60)
print("HYPOTHESIS TESTING REVIEW")
print("=" * 60)

# --- Hypothesis Q1 ---
print("\n--- Hypothesis Q1 ---")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print("t-statistic:", t_stat)
print("p-value:", p_value)

# --- Hypothesis Q2 ---
print("\n--- Hypothesis Q2 ---")
if p_value < 0.05:
    print("The result is statistically significant at alpha = 0.05.")
else:
    print("The result is not statistically significant at alpha = 0.05.")

# --- Hypothesis Q3 ---
print("\n--- Hypothesis Q3 ---")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
paired_t, paired_p = stats.ttest_rel(before, after)
print("Paired t-test t-statistic:", paired_t)
print("Paired t-test p-value:", paired_p)

# --- Hypothesis Q4 ---
print("\n--- Hypothesis Q4 ---")
scores = [72, 68, 75, 70, 69, 74, 71, 73]
one_sample_t, one_sample_p = stats.ttest_1samp(scores, popmean=70)
print("One-sample t-statistic:", one_sample_t)
print("One-sample p-value:", one_sample_p)

# --- Hypothesis Q5 ---
print("\n--- Hypothesis Q5 ---")
one_tailed_test = stats.ttest_ind(group_a, group_b, alternative="less")
print("One-tailed p-value:", one_tailed_test.pvalue)

# --- Hypothesis Q6 ---
print("\n--- Hypothesis Q6 ---")
if p_value < 0.05:
    print("Group A scores appear lower than Group B scores, and the difference is unlikely due to chance.")
else:
    print("The difference between Group A and Group B may be due to chance.")


print("\n" + "=" * 60)
print("CORRELATION REVIEW")
print("=" * 60)

# --- Correlation Q1 ---
print("\n--- Correlation Q1 ---")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)
print("Correlation coefficient:", corr_matrix[0, 1])

# I expect a perfect positive correlation because y increases exactly with x.

# --- Correlation Q2 ---
print("\n--- Correlation Q2 ---")
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]
r, p = pearsonr(x, y)
print("Pearson r:", r)
print("p-value:", p)

# --- Correlation Q3 ---
print("\n--- Correlation Q3 ---")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}
df_people = pd.DataFrame(people)
corr_people = df_people.corr()
print(corr_people)

# --- Correlation Q4 ---
print("\n--- Correlation Q4 ---")
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.figure()
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- Correlation Q5 ---
print("\n--- Correlation Q5 ---")
plt.figure()
sns.heatmap(corr_people, annot=True)
plt.title("Correlation Heatmap")
plt.show()


print("\n" + "=" * 60)
print("PIPELINES")
print("=" * 60)

# --- Pipeline Q1 ---
print("\n--- Pipeline Q1 ---")
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }

def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

result = data_pipeline(arr)
for key, value in result.items():
    print(f"{key}: {value}")