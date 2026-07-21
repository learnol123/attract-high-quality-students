# =========================================
# AUTO INSTALL + IMPORT (INCLUDING TKINTER)
# =========================================
import subprocess
import sys

def install_and_import(package, import_name=None):
    try:
        if import_name:
            __import__(package)
        else:
            __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install_and_import("pandas")
install_and_import("numpy")
install_and_import("scikit-learn", "sklearn")
install_and_import("xgboost")
install_and_import("openpyxl") 
install_and_import("matplotlib") 
install_and_import("seaborn")

# =========================================
# TKINTER FILE PICKER
# =========================================
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    raise ImportError("Tkinter is not installed. Please install it manually.")

# Hide root window
root = tk.Tk()
root.withdraw()

# Open file dialog
file_path = filedialog.askopenfilename(
    title="Select your Excel file",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 2. LOAD DATA
# =========================================
df = pd.read_excel(file_path)
print("File loaded:", file_path)

df.columns = df.columns.str.strip()

# =========================================
# 3. CLEANING & FORMATTING
# =========================================
df = df.copy()

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[()]", "", regex=True)
)

df = df.drop_duplicates()

# --- Numeric cleaning ---
if "qs_ranking" in df.columns:
    df["qs_ranking"] = df["qs_ranking"].astype(str).str.extract(r'(\d+)')[0]
    df["qs_ranking"] = pd.to_numeric(df["qs_ranking"], errors="coerce")

if "avg_salary_cad" in df.columns:
    df["avg_salary_cad"] = pd.to_numeric(
        df["avg_salary_cad"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    )

if "tuition_int'l_cad" in df.columns:
    df["tuition_int'l_cad"] = pd.to_numeric(
        df["tuition_int'l_cad"].astype(str).str.replace(",", "").str.replace("$", "").str.strip(),
        errors="coerce"
    )

if "employment_rate" in df.columns:
    df["employment_rate"] = pd.to_numeric(
        df["employment_rate"].astype(str).str.replace("%", "").str.strip(),
        errors="coerce"
    )
    df["employment_rate"] = df["employment_rate"].apply(
        lambda x: x / 100 if pd.notnull(x) and x > 1 else x
    )

if "living_cost_/yr_transformed" in df.columns:
    df["living_cost_/yr_transformed"] = pd.to_numeric(
        df["living_cost_/yr_transformed"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    )

if "duration_in_months" in df.columns:
    df["duration_in_months"] = pd.to_numeric(df["duration_in_months"], errors="coerce")

# Drop rows where target variable (tuition) is missing
df = df.dropna(subset=["tuition_int'l_cad"])

# --- Missing values ---
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

for col in df.columns:
    if df[col].dtype == 'object' or isinstance(df[col].dtype, CategoricalDtype):
        df[col] = df[col].fillna('Unknown')

# --- Outliers handling ---
for col in numeric_cols:
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q1, q99)

# =========================================
# 4. FEATURE ENGINEERING & TARGET SETTING
# =========================================

# --- Dependent Variable (Y) ---
y = df["tuition_int'l_cad"]

# --- Derived Features ---
if "qs_ranking" in df.columns:
    df["top_uni_flag"] = (df["qs_ranking"] < 100).astype(int)

if "coop_level" in df.columns:
    df["coop_flag"] = (df["coop_level"] > 0).astype(int)

# --- Categorize Tuition Tiers (For Reporting) ---
def categorize_tuition(fee):
    if fee >= df['tuition_int\'l_cad'].quantile(0.66):
        return 'High'
    elif fee >= df['tuition_int\'l_cad'].quantile(0.33):
        return 'Medium'
    return 'Low'

df['tuition_tier'] = df['tuition_int\'l_cad'].apply(categorize_tuition)

# =========================================
# 5. INDEPENDENT FEATURES (X) - DYNAMIC SELECTION
# =========================================
drop_patterns = [
    r'tuition', r'cost', r'price', r'fee', r'tier', r'score', r'attractiveness',
    r'university', r'program', r'school', r'url', r'link', r'http', r'https', 
    r'institution', r'id\b', r'label', r'name', r'report', r'html', r'pdf', r'page', r'www'
]

combined_pattern = "|".join(drop_patterns)

# Dynamically identify columns to drop from feature set X
cols_to_drop = [
    col for col in df.columns 
    if re.search(combined_pattern, col, re.IGNORECASE)
]

# Keep all non-matching program features for model training
X = df.drop(columns=cols_to_drop, errors="ignore")

# Drop zero-variance columns
X = X.loc[:, X.nunique() > 1]

# Cast categorical types to string for pipeline processing
for col in X.select_dtypes(include=['object', 'string', 'category']).columns:
    X[col] = X[col].astype(str)

categorical_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object', 'string', 'category']).columns.tolist()

print("\n--- Features Dynamically Discovered by Model (X) ---")
print("Numeric Features:", numeric_cols)
print("Categorical Features:", categorical_cols)

# =========================================
# 6. PIPELINE SETUP
# =========================================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# =========================================
# 7. TRAIN-TEST SPLIT & EVALUATION
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\n=========================================")
print(" MODEL PERFORMANCE (International Tuition) ")
print("=========================================")
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE (CAD):", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE (CAD):", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# Classification evaluation on predicted vs actual tuition tiers
y_pred_tier = pd.Series(y_pred, index=y_test.index).apply(categorize_tuition)
y_test_tier = y_test.apply(categorize_tuition)

print("\nTUITION TIER CLASSIFICATION PERFORMANCE:")
print("Accuracy:", round(accuracy_score(y_pred_tier, y_test_tier), 3))
print("\nClassification Report:\n", classification_report(y_pred_tier, y_test_tier))

# =========================================
# 8. CROSS VALIDATION
# =========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=kf)

print("\nCROSS VALIDATION PERFORMANCE:")
print("CV R2 Scores:", np.round(scores, 3))
print("Avg CV R2 Score:", round(scores.mean(), 3))

# =========================================
# 9. SAVE OUTPUT
# =========================================
df["predicted_tuition_cad"] = pipeline.predict(X)

id_cols = [c for c in ["university", "program"] if c in df.columns]
meta_df = df[id_cols].copy() if len(id_cols) > 0 else pd.DataFrame(index=df.index)

combined_df = pd.concat([
    meta_df,
    X,
    y.rename("actual_tuition_int'l_cad"),
    df["predicted_tuition_cad"],
    df['tuition_tier']
], axis=1)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"predicted_tuition_output_{timestamp}.xlsx"
combined_df.to_excel(output_filename, index=False)

abs_path = os.path.abspath(output_filename)
print(f"\nSaved predicted dataset as: {abs_path}")

try:
    os.startfile(output_filename)
except Exception as e:
    print(f"Could not open automatically: {e}")

# =========================================
# 10. VISUALIZATIONS
# =========================================
sns.set_theme(style="whitegrid")

# --- Graph 1: Simple Windsor vs Top Programs Comparison ---
if "university" in df.columns and "program" in df.columns:
    df["label"] = df["university"] + " | " + df["program"]
    top_programs = df.sort_values("tuition_int'l_cad", ascending=False).head(5)
    windsor_programs = df[df["university"].str.contains("Windsor", case=False)]

    simple_compare = pd.concat([top_programs, windsor_programs]).drop_duplicates()
    simple_compare = simple_compare.sort_values("tuition_int'l_cad", ascending=True)

    colors = ["#E53935" if "Windsor" in str(uni) else "#1E88E5" for uni in simple_compare["university"]]

    plt.figure(figsize=(10, 5))
    plt.barh(simple_compare["label"], simple_compare["tuition_int'l_cad"], color=colors)

    for i, v in enumerate(simple_compare["tuition_int'l_cad"]):
        plt.text(v + 500, i, f"${v:,.0f}", va='center', fontsize=10, fontweight='bold')

    plt.title("Tuition Comparison: University of Windsor vs Top Programs", fontsize=12, fontweight='bold')
    plt.xlabel("International Tuition Fee (CAD)", fontsize=10)
    plt.tight_layout()
    plt.show()

# --- Graph 2: Top Key Factors Driving Tuition Price (Aggregated Parent Columns) ---
ohe = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
encoded_cat_cols = list(ohe.get_feature_names_out(categorical_cols)) if len(categorical_cols) > 0 else []

feature_parent_map = {col: col for col in numeric_cols}
for cat_col in categorical_cols:
    for enc_col in encoded_cat_cols:
        if enc_col.startswith(f"{cat_col}_"):
            feature_parent_map[enc_col] = cat_col

importances = pipeline.named_steps["model"].feature_importances_
all_feature_names = numeric_cols + encoded_cat_cols

aggregated_imp = {}
for feat, imp in zip(all_feature_names, importances):
    parent = feature_parent_map.get(feat, feat)
    aggregated_imp[parent] = aggregated_imp.get(parent, 0.0) + imp

formatted_imp = {
    k.replace("_", " ").title(): v 
    for k, v in aggregated_imp.items()
}

top_imp = pd.Series(formatted_imp).sort_values(ascending=False).head(5)

plt.figure(figsize=(9, 4.5))
ax = sns.barplot(x=top_imp.values, y=top_imp.index, palette="mako")

for i, v in enumerate(top_imp.values):
    ax.text(v + 0.002, i, f"{v*100:.1f}%", va='center', fontsize=10, fontweight='bold')

plt.title("Top Key Factors Driving Tuition Price (Aggregated Drivers)", fontsize=12, fontweight='bold')
plt.xlabel("Relative Importance Score (%)", fontsize=10)
plt.ylabel("Core Program Drivers", fontsize=10)
plt.tight_layout()
plt.show()

# --- Graph 3: RADAR CHART (University of Windsor vs Competitor Average) ---
if "university" in df.columns:
    radar_cols_map = {
        "tuition_int'l_cad": "Tuition Fee",
        "qs_ranking": "QS Rank Standing",
        "duration_in_months": "Program Duration",
        "avg_salary_cad": "Avg Salary",
        "employment_rate": "Employment Rate"
    }

    avail_radar_cols = [c for c in radar_cols_map.keys() if c in df.columns]

    if len(avail_radar_cols) >= 3:
        radar_df = df[avail_radar_cols].copy()

        scaler = MinMaxScaler(feature_range=(10, 100))
        scaled_mat = scaler.fit_transform(radar_df)
        scaled_df = pd.DataFrame(scaled_mat, columns=avail_radar_cols, index=df.index)

        if "qs_ranking" in scaled_df.columns:
            scaled_df["qs_ranking"] = 110 - scaled_df["qs_ranking"]

        scaled_df["university"] = df["university"]

        windsor_mask = scaled_df["university"].str.contains("Windsor", case=False, na=False)
        windsor_scores = scaled_df[windsor_mask][avail_radar_cols].mean()
        competitor_scores = scaled_df[~windsor_mask][avail_radar_cols].mean()

        labels = [radar_cols_map[c] for c in avail_radar_cols]
        num_vars = len(labels)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        w_vals = windsor_scores.values.tolist() + [windsor_scores.values[0]]
        c_vals = competitor_scores.values.tolist() + [competitor_scores.values[0]]
        radar_angles = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        ax.plot(radar_angles, w_vals, color='#E53935', linewidth=2.5, label='University of Windsor')
        ax.fill(radar_angles, w_vals, color='#E53935', alpha=0.25)

        ax.plot(radar_angles, c_vals, color='#1E88E5', linewidth=2.5, label='Competitor Average')
        ax.fill(radar_angles, c_vals, color='#1E88E5', alpha=0.2)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')

        plt.title("Competitive Benchmark: University of Windsor vs Competitors", y=1.08, fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.show()

# --- Graph 4: NUMBER OF PROGRAMS PER TUITION TIER (COLOR CODED) ---
tier_order = ['Low', 'Medium', 'High']
tier_counts = df['tuition_tier'].value_counts().reindex(tier_order, fill_value=0)

tier_colors = {
    'Low': '#0D47A1',     # Deep Blue
    'Medium': '#F57F17',  # Amber Gold
    'High': '#B71C1C'     # Dark Red
}
palette = [tier_colors[tier] for tier in tier_order]

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=tier_counts.index, y=tier_counts.values, palette=palette)

# Dynamic fee range annotations above each bar
q33 = df["tuition_int'l_cad"].quantile(0.33)
q66 = df["tuition_int'l_cad"].quantile(0.66)

fee_ranges = {
    'Low': f"< ${q33:,.0f}",
    'Medium': f"${q33:,.0f} - ${q66:,.0f}",
    'High': f"> ${q66:,.0f}"
}

for i, (tier, count) in enumerate(tier_counts.items()):
    fee_label = fee_ranges[tier]
    ax.text(i, count + 0.3, f"{count} Programs\n({fee_label})", 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Program Distribution by Tuition Tier (Low, Medium, High)", fontsize=12, fontweight='bold')
plt.xlabel("Tuition Fee Category Tier", fontsize=10)
plt.ylabel("Number of Programs", fontsize=10)
plt.ylim(0, max(tier_counts.values) * 1.25)
plt.tight_layout()
plt.show()

# --- Graph 5: Simple Model Accuracy (Actual vs Predicted) ---
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="#1A237E", alpha=0.7, s=70, label="Programs Evaluated")

min_p = min(y_test.min(), y_pred.min())
max_p = max(y_test.max(), y_pred.max())
plt.plot([min_p, max_p], [min_p, max_p], 'r--', lw=2, label="100% Accurate Line")

plt.title("Model Prediction Accuracy", fontsize=12, fontweight='bold')
plt.xlabel("Actual Tuition (CAD)", fontsize=10)
plt.ylabel("Predicted Tuition (CAD)", fontsize=10)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()