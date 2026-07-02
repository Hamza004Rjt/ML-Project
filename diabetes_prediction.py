"""
Smart Healthcare Prediction System — Diabetes Risk Assessment
=============================================================
Dataset : BRFSS / CDC Diabetes Health Indicators
          (Diabetes_012_health_indicators_BRFSS2015)

Columns : Diabetes_012 (target), HighBP, HighChol, CholCheck, BMI,
          Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits,
          Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
          GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
          Education, Income

Target   : 3-class → 0 = Non-Diabetic  |  1 = Pre-Diabetic  |  2 = Diabetic
Models   : Random Forest · Gradient Boosting · AdaBoost · Voting Classifier
"""

# ─────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────
import warnings, os
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
OUTPUT_DIR   = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD DATASET
# ─────────────────────────────────────────────
def load_data(filepath: str = "diabetic_data.csv") -> pd.DataFrame:
    """
    Load the BRFSS Diabetes Health Indicators dataset.
    Expected columns: Diabetes_012, HighBP, HighChol, CholCheck, BMI,
    Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
    HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth,
    PhysHlth, DiffWalk, Sex, Age, Education, Income
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Place 'diabetic_data.csv' in the same directory as this script."
        )
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded dataset  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────
TARGET_COL   = "Diabetes_012"
CLASS_NAMES  = ["Non-Diabetic", "Pre-Diabetic", "Diabetic"]

# All feature columns (everything except the target)
FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline → returns X (array), y (array), feature_names.

    The BRFSS dataset is already fully numeric with no missing values,
    so only imputation (safety) + scaling are required.
    """

    # ── 2.1  Validate required columns ────────────────────────────────────
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    # ── 2.2  Extract target ────────────────────────────────────────────────
    y = df[TARGET_COL].astype(int).values      # 0 / 1 / 2  (already encoded)

    # ── 2.3  Extract features ──────────────────────────────────────────────
    X_df = df[FEATURE_COLS].copy()

    # ── 2.4  Impute any residual NaNs (robustness guard) ──────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_df)

    # ── 2.5  Scale features ────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    print(f"[✓] Preprocessing done → X: {X_scaled.shape}  |  Classes: {np.bincount(y)}")
    return X_scaled, y, FEATURE_COLS


# ─────────────────────────────────────────────
# 3.  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_models():
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    gb  = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=RANDOM_STATE
    )
    ada = AdaBoostClassifier(
        n_estimators=150, learning_rate=0.5, random_state=RANDOM_STATE
    )
    vote = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("ada", ada)],
        voting="soft", n_jobs=-1
    )
    return {
        "Random Forest"     : rf,
        "Gradient Boosting" : gb,
        "AdaBoost"          : ada,
        "Voting Classifier" : vote,
    }


# ─────────────────────────────────────────────
# 4.  TRAINING & EVALUATION
# ─────────────────────────────────────────────
def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    results = []
    trained = {}

    for name, model in models.items():
        print(f"  Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model"    : name,
            "Accuracy" : round(acc  * 100, 2),
            "Precision": round(prec * 100, 2),
            "Recall"   : round(rec  * 100, 2),
            "F1-Score" : round(f1   * 100, 2),
        })
        trained[name] = (model, y_pred)
        print(f"Acc={acc*100:.2f}%  F1={f1*100:.2f}%")

    df_results = pd.DataFrame(results).set_index("Model")
    return df_results, trained


# ─────────────────────────────────────────────
# 5.  PLOTS
# ─────────────────────────────────────────────
def plot_comparison_table(df_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText  = df_results.values,
        colLabels = df_results.columns,
        rowLabels = df_results.index,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#2E75B6")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#EAF2FB" if r % 2 == 0 else "white")

    plt.title("Model Comparison — Accuracy · Precision · Recall · F1-Score (%)",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved → {path}")


def plot_metrics_bar(df_results: pd.DataFrame):
    ax = df_results.plot(
        kind="bar", figsize=(12, 5), colormap="Set2",
        edgecolor="grey", linewidth=0.5
    )
    ax.set_title("Ensemble Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_xticklabels(df_results.index, rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in ax.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center", va="bottom", fontsize=7.5
        )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "metrics_bar_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved → {path}")


def plot_confusion_matrix(model_name: str, y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved → {path}")


def plot_feature_importance(rf_model, feature_names, top_n=20):
    # Cap top_n to the actual number of features
    top_n = min(top_n, len(feature_names))
    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color="#2E75B6", edgecolor="grey")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved → {path}")


# ─────────────────────────────────────────────
# 6.  HEALTH RECOMMENDATIONS
# ─────────────────────────────────────────────
RECOMMENDATIONS = {
    0: (
        "✅ NON-DIABETIC\n"
        "   • Maintain a balanced diet rich in vegetables, whole grains, and lean proteins.\n"
        "   • Exercise at least 150 min/week (e.g., brisk walking, cycling).\n"
        "   • Schedule annual blood glucose screenings.\n"
        "   • Avoid smoking and limit alcohol intake."
    ),
    1: (
        "⚠️  PRE-DIABETIC\n"
        "   • Reduce refined sugars and carbohydrates; increase fibre intake.\n"
        "   • Aim for 5–7 % body-weight loss if overweight.\n"
        "   • Monitor blood glucose every 3–6 months.\n"
        "   • Consult a dietitian for a personalised meal plan.\n"
        "   • Consider Metformin if lifestyle changes are insufficient."
    ),
    2: (
        "🚨 DIABETIC\n"
        "   • Strictly follow prescribed medication regimen (insulin / oral agents).\n"
        "   • Test blood glucose at home 2–4 times daily.\n"
        "   • Follow a low-glycaemic-index diet; limit portion sizes.\n"
        "   • Schedule quarterly HbA1c tests and annual eye / kidney / foot exams.\n"
        "   • Engage in regular moderate exercise under medical supervision."
    ),
}

def print_recommendation(prediction: int):
    print("\n" + "═" * 55)
    print(" HEALTH RECOMMENDATION")
    print("═" * 55)
    print(RECOMMENDATIONS[prediction])
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# 7.  SINGLE-PATIENT PREDICTION DEMO
# ─────────────────────────────────────────────
def demo_prediction(model, feature_dim: int):
    """
    Simulates a real-time prediction for one patient using random scaled values.
    In production this would accept form inputs from a web interface.
    """
    rng    = np.random.default_rng(seed=7)
    sample = rng.standard_normal((1, feature_dim))   # already in scaled space
    pred   = model.predict(sample)[0]
    prob   = model.predict_proba(sample)[0]

    print("\n" + "─" * 55)
    print(" DEMO: Single-Patient Prediction")
    print("─" * 55)
    print(f"  Predicted Class  : {CLASS_NAMES[pred]}  (class {pred})")
    for i, cn in enumerate(CLASS_NAMES):
        print(f"  P({cn:>14s}) = {prob[i]*100:5.1f}%")
    print_recommendation(pred)


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Smart Healthcare Prediction System — Diabetes Risk")
    print("=" * 60 + "\n")

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_data("diabetic_data.csv")

    # ── Preprocess ────────────────────────────────────────────────────────
    print("\n[STEP 1] Preprocessing …")
    X, y, feature_names = preprocess(df)

    # ── Train/Test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[✓] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ── Build & Evaluate ──────────────────────────────────────────────────
    print("\n[STEP 2] Training & evaluating ensemble models …")
    models          = build_models()
    df_res, trained = evaluate_models(models, X_train, X_test, y_train, y_test)

    # ── Comparison Table ──────────────────────────────────────────────────
    print("\n[STEP 3] Results Summary")
    print("─" * 55)
    print(df_res.to_string())
    print("─" * 55)

    best_name = df_res["F1-Score"].idxmax()
    print(f"\n[★] Best model by F1-Score: {best_name}")

    # ── Classification Report ─────────────────────────────────────────────
    best_model, best_pred = trained[best_name]
    print(f"\n[STEP 4] Detailed Report for: {best_name}")
    print(classification_report(y_test, best_pred, target_names=CLASS_NAMES))

    # ── Plots ─────────────────────────────────────────────────────────────
    print("[STEP 5] Generating plots …")
    plot_comparison_table(df_res)
    plot_metrics_bar(df_res)
    plot_confusion_matrix(best_name, y_test, best_pred)

    rf_model = trained["Random Forest"][0]
    plot_feature_importance(rf_model, feature_names)

    # Confusion matrices for remaining models
    for name, (_, pred) in trained.items():
        if name != best_name:
            plot_confusion_matrix(name, y_test, pred)

    # ── Demo Prediction ───────────────────────────────────────────────────
    print("[STEP 6] Running demo prediction …")
    demo_prediction(best_model, X_test.shape[1])   # removed unused scaler_placeholder arg

    print(f"\n[✓] All plots saved to './{OUTPUT_DIR}/' directory.")
    print("=" * 60)
    print("  Pipeline complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
