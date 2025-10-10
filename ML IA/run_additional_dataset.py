
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, classification_report)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


paths = [
    r"D:\ML IA\ML IA\data\credit_card_defaulter (1).csv", 
    
]
df = None
for p in paths:
    try:
        df = pd.read_csv(p)
        print(f"Loaded: {p}")
        break
    except Exception:
        pass
if df is None:
    raise FileNotFoundError("Additional dataset CSV not found at expected paths.")

print("Initial shape:", df.shape)
print("Columns (first 30):", df.columns.tolist()[:30])

for junk in ["Unnamed: 0", "Unnamed: 0.1", "index"]:
    if junk in df.columns:
        df = df.drop(columns=[junk])
        print(f"Dropped junk column: {junk}")


possible_targets = ["Default_Payment_Next_Month","default.payment.next.month",
                    "default_payment_next_month","default","DEFAULT","is_defaulter","target"]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    target_col = df.columns[-1]
print("Using target column:", target_col)

if df[target_col].dtype == object:
    tmp = df[target_col].astype(str).str.strip().str.lower()
    mapped = tmp.map({
        "yes": 1, "no": 0, "y": 1, "n": 0,
        "true": 1, "false": 0, "1": 1, "0": 0
    })
    df[target_col] = mapped

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)


cat_cols = []
for c in df.columns:
    if c == target_col:
        continue
    cl = c.lower()
    if cl in ["student","sex","education","marriage"] or cl.startswith("pay"):
        cat_cols.append(c)
cat_cols = sorted(set(cat_cols))
print("Categorical-like columns detected:", cat_cols)

for c in cat_cols:
    df[c] = df[c].astype('category')
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df = df.fillna(df.median(numeric_only=True))

X = df.drop(columns=[target_col])
y = df[target_col]


corrs = {}
for col in X.columns:
    try:
        corrs[col] = abs(np.corrcoef(X[col].astype(float), y.astype(float))[0,1])
    except Exception:
        corrs[col] = 0.0
k = min(15, X.shape[1])
top_features = sorted(corrs, key=lambda x: corrs[x], reverse=True)[:k]
print(f"Selected top-{k} features:", top_features)
X = X[top_features]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "LogisticRegression": LogisticRegression(solver='liblinear', random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}

metrics_table = []
probas = {}
preds_store = {}

for name, model in models.items():
    print("\nTraining:", name)
    if name == "LogisticRegression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    metrics_table.append([name, acc, prec, rec, f1, auc])
    if y_proba is not None:
        probas[name] = y_proba
    preds_store[name] = y_pred

os.makedirs(".", exist_ok=True)


if len(np.unique(y_test)) == 2 and len(probas) > 0:
    plt.figure()
    for name, y_proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Additional Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_additional.png", dpi=200)
    plt.show()


for name, y_pred in preds_store.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix — {name} (Additional)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(f"cm_additional_{name}.png", dpi=200)
    plt.show()


metrics_df = pd.DataFrame(metrics_table, columns=["Model","Accuracy","Precision","Recall","F1","AUC"])
plt.figure()
x = np.arange(len(metrics_df["Model"]))
width = 0.25
plt.bar(x - width, metrics_df["Accuracy"], width, label="Accuracy")
plt.bar(x,          metrics_df["F1"],       width, label="F1")
plt.bar(x + width,  metrics_df["AUC"],      width, label="AUC")
plt.xticks(x, metrics_df["Model"])
plt.ylim(0, 1)
plt.title("Model Comparison — Additional Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("compare_additional.png", dpi=200)
plt.show()

print("\nSaved: roc_additional.png, cm_additional_<Model>.png, compare_additional.png")
