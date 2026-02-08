import os
import snowflake.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from dotenv import load_dotenv


load_dotenv()

print("🔌 Connecting to Snowflake securely...")

try:
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOW_USER'),
        password=os.getenv('SNOW_PASS'),
        account=os.getenv('SNOW_ACCOUNT'),
        warehouse='COMPUTE_WH',
        database='BICEP_CURL',
        schema='PUBLIC'
    )
except:
    raise Exception("No credentials found")
    


# --- 2. FETCH DATA ---
print("☁️ Downloading training data...")
cursor = ctx.cursor()
cursor.execute('SELECT * FROM "TEMP"')
data = cursor.fetch_pandas_all()
ctx.close()
print(f"✅ Downloaded {len(data)} rows.")

# --- 3. CLEAN & PREPARE ---
target_col = 'LABEL' if 'LABEL' in data.columns else 'label'
X = data.drop(columns=[target_col])
y = data[target_col]

# Standardize feature names to UPPERCASE
X.columns = [c.upper() for c in X.columns]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. TRAIN MODEL ---
print("🧠 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,      # More trees = better stability
    max_depth=None,        # Let trees grow fully
    random_state=42,
    class_weight='balanced'  # Handle imbalance
)
model.fit(X_train, y_train)

# --- 5. EVALUATE ---
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("------------------------------------------------")
print(f"🏆 MODEL ACCURACY: {acc*100:.2f}%")
print("------------------------------------------------")
print(classification_report(y_test, pred))

# --- 6. SAVE MODEL + SCALER + FEATURE LIST ---
joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': X.columns.tolist()
}, "bicep_model_v2.pkl")
print("💾 Saved 'bicep_model_v2.pkl'. Ready for live demo!")
