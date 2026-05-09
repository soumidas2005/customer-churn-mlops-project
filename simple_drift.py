import pandas as pd
from scipy.stats import ks_2samp

print("🚀 Running Simple Data Drift Check...")

# Load data
old_data = pd.read_csv("final_dataset.csv")
new_data = pd.read_csv("final_dataset.csv")

# Keep only numeric
old_data = old_data.select_dtypes(include=["number"])
new_data = new_data.select_dtypes(include=["number"])

# Match columns
common_cols = list(set(old_data.columns) & set(new_data.columns))

results = []

for col in common_cols:
    stat, p_value = ks_2samp(old_data[col], new_data[col])

    drift = "Drift" if p_value < 0.05 else "No Drift"

    results.append([col, p_value, drift])

    print(f"{col} → {drift} (p={p_value:.4f})")

# -----------------------------
# CREATE HTML REPORT
# -----------------------------
html_content = "<h1>Data Drift Report</h1><table border='1'><tr><th>Feature</th><th>p-value</th><th>Status</th></tr>"

for row in results:
    html_content += f"<tr><td>{row[0]}</td><td>{row[1]:.4f}</td><td>{row[2]}</td></tr>"

html_content += "</table>"

with open("data_drift_report.html", "w") as f:
    f.write(html_content)

print("\n✅ HTML Report Created: data_drift_report.html")