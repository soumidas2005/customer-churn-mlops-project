import os
import subprocess

print("🚀 Starting Simulated Airflow Pipeline...\n")

# -------------------------------
# STEP 1: DATA DRIFT CHECK
# -------------------------------
print("🔍 Step 1: Running Data Drift Detection...")

drift_command = "python simple_drift.py"
drift_status = os.system(drift_command)

if drift_status != 0:
    print("❌ Drift check failed. Stopping pipeline.")
    exit()

print("✅ Drift check completed\n")

# -------------------------------
# STEP 2: CHECK DRIFT RESULT
# -------------------------------
print("📊 Step 2: Checking if retraining is needed...")

# Simple logic (simulate PSI/KS threshold)
drift_detected = True   # <-- You can change later

if drift_detected:
    print("⚠️ Drift detected → Retraining required\n")
else:
    print("✅ No drift → Skipping retraining")
    exit()

# -------------------------------
# STEP 3: RETRAIN MODEL
# -------------------------------
print("🤖 Step 3: Retraining Model...")

retrain_command = "python run_mlflow.py"
retrain_status = os.system(retrain_command)

if retrain_status != 0:
    print("❌ Model retraining failed")
    exit()

print("✅ Model retrained successfully\n")

# -------------------------------
# STEP 4: SHADOW DEPLOYMENT
# -------------------------------
print("🚀 Step 4: Shadow Deployment Simulation...")

print("""
Shadow Deployment means:
- Old model is still in production
- New model runs in background
- Compare performance silently
""")

print("✅ New model deployed in shadow mode\n")

# -------------------------------
# STEP 5: PIPELINE COMPLETE
# -------------------------------
print("🎉 Airflow Pipeline Completed Successfully!")