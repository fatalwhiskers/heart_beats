import pandas as pd
from pathlib import Path

csv_path = r"outputs\FFT_2002.csv"
data = pd.read_csv(csv_path)

best_by_mae = data.loc[
    data.groupby("ROI")["MAE (bpm)"].idxmin(),
    ["ROI", "Extraction Method", "MAE (bpm)", "RMSE (bpm)", "Pearson r"]
]

best_by_rmse = data.loc[
    data.groupby("ROI")["RMSE (bpm)"].idxmin(),
    ["ROI", "Extraction Method", "MAE (bpm)", "RMSE (bpm)", "Pearson r"]
]

print("=== Best by MAE per ROI ===")
print(best_by_mae.to_string(index=False))

print("\n=== Best by RMSE per ROI ===")
print(best_by_rmse.to_string(index=False))
