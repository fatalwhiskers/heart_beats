import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

fft_df = pd.read_csv("fft.csv")
fft_df["Frequency Method"] = "FFT"

welch_df = pd.read_csv("welch.csv")
welch_df["Frequency Method"] = "Welch"

results_df = pd.concat([fft_df, welch_df], ignore_index=True)

by_method_subject = results_df.groupby(["Frequency Method", "Subject ID"]).agg({
    "MAE (bpm)": ["mean", "std"],
    "RMSE (bpm)": ["mean", "std"],
    "Pearson r": ["mean", "std"],
    "Bias (bpm)": ["mean", "std"],
    "SD (bpm)": ["mean", "std"],
    "LoA Lower (bpm)": ["mean"],
    "LoA Upper (bpm)": ["mean"]
}).round(2)

print("Summary Statistics by Method and Subject:")
print(by_method_subject)

overall_by_method = results_df.groupby("Frequency Method").agg({
    "MAE (bpm)": ["mean", "std", "min", "max"],
    "RMSE (bpm)": ["mean", "std"],
    "Pearson r": ["mean", "std"],
    "Bias (bpm)": ["mean", "std"],
    "SD (bpm)": ["mean", "std"]
}).round(2)

print("\nOverall Summary Across All Subjects:")
print(overall_by_method)

mae_pivot = results_df.pivot_table(
    index=["Subject ID", "Recording ID", "Signal"],
    columns="Frequency Method",
    values="MAE (bpm)"
)

t_stat, p_val = stats.ttest_rel(mae_pivot["FFT"], mae_pivot["Welch"], nan_policy="omit")
print(f"\nPaired t-test for MAE (FFT vs Welch): t={t_stat:.2f}, p={p_val:.4f}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x="Frequency Method", y="MAE (bpm)")
plt.title("MAE Comparison Across Methods (All Subjects)")
plt.ylabel("MAE (bpm)")
plt.xlabel("Method")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Subject ID", y="MAE (bpm)", hue="Frequency Method")
plt.title("Mean MAE by Subject and Method")
plt.ylabel("MAE (bpm)")
plt.xlabel("Subject ID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fft_method_df = results_df[results_df["Frequency Method"] == "FFT"]
plt.figure(figsize=(10, 6))
plt.scatter(fft_method_df["Mean Ground Truth (bpm)"], fft_method_df["Mean Error (bpm)"], alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.title("Bland-Altman: Errors vs Ground Truth (FFT Method)")
plt.xlabel("Mean Ground Truth HR (bpm)")
plt.ylabel("Mean Error (bpm)")
plt.tight_layout()
plt.show()

r_pivot = results_df.pivot_table(index="Signal", columns="Frequency Method", values="Pearson r")
plt.figure(figsize=(10, 6))
sns.heatmap(r_pivot, annot=True, cmap="coolwarm", center=0)
plt.title("Average Pearson r by Signal and Method")
plt.xlabel("Method")
plt.ylabel("Signal")
plt.tight_layout()
plt.show()
