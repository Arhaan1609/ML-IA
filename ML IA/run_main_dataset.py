

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
    r"D:\ML IA\ML IA\data\default of credit card clients.xls"
    
]
df = None
for p in paths:
    try:
        df = pd.read_excel(p)
        print(f"Loaded: {p}")
        break
    except Exception:
        pass
if df is None:
    raise FileNotFoundError("Could not load dataset from expected paths.")

print("Initial shape:", df.shape)
print("Columns (first 30):", df.columns.tolist()[:30])

generic_uci = set(["Y","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23"])
if set(df.columns) & generic_uci == set(df.columns):
    for p in paths:
        try:
            df = pd.read_excel(p, header=1)
            print(f"Re-loaded with header=1: {p}")
            break
        except Exception:
            pass

if set(df.columns) & generic_uci == set(df.columns):
    rename_map = {
        "Unnamed: 0": "ID",
        "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE", "X5": "AGE",
        "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4", "X10": "PAY_5", "X11": "PAY_6",
        "X12": "BILL_AMT1", "X13": "BILL_AMT2", "X14": "BILL_AMT3", "X15": "BILL_AMT4",
        "X16": "BILL_AMT5", "X17": "BILL_AMT6",
        "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3", "X21": "PAY_AMT4",
        "X22": "PAY_AMT5", "X23": "PAY_AMT6",
        "Y": "default payment next month",
    }
    df = df.rename(columns=rename_map)
    print("Renamed UCI-style headers to canonical names.")

for junk in ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 1"]:
    if junk in df.columns:
        df = df.drop(columns=[junk])
        print(f"Dropped junk column: {junk}")

print("Final columns (first 30):", df.columns.tolist()[:30])


for id_col in ["ID", "Id", "id", "Id_x", "ID_x"]:
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        print(f"Dropped id column: {id_col}")

possible_targets = [
    "default payment next month",
    "Default_Payment_Next_Month",
    "default.payment.next.month",
    "default_payment_next_month",
    "default",
]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    target_col = df.columns[-1]
    print(f"Target not found among expected names; using last column: {target_col}")
else:
    print(f"Found target column: {target_col}")

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
if df[target_col].isna().all():
    raise ValueError(f"Target column '{target_col}' could not be parsed to numbers.")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

cat_cols = []
for c in df.columns:
    if c == target_col:
        continue
    cl = c.lower()
    if c.upper() in ["SEX","EDUCATION","MARRIAGE"] or cl.startswith("pay_") or cl == "pay_0":
        cat_cols.append(c)
cat_cols = sorted(set(cat_cols))
print("Categorical-like columns:", cat_cols)

for c in cat_cols:
    df[c] = df[c].astype("category")
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
    plt.title("ROC Curve — Main Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_main.png", dpi=200)
    plt.show()


for name, y_pred in preds_store.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix — {name} (Main)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(f"cm_main_{name}.png", dpi=200)
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
plt.title("Model Comparison — Main Dataset")
plt.legend()
plt.tight_layout()
plt.savefig("compare_main.png", dpi=200)
plt.show()

print("\nSaved: roc_main.png, cm_main_<Model>.png, compare_main.png")
