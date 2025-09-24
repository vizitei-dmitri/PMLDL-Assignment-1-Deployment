import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np

RANDOM_STATE = 83
def main():
    df = pd.read_csv("data/raw/bank.csv")
    yn = {"yes": True, "no": False}

    for col in ["housing", "loan"]:
        if df[col].dtype == object:
            df[col] = df[col].map(yn)
    
    y = (df["deposit"] == "yes").astype(int)

    features = ["age", "job", "marital", "education", "balance",
                "housing", "loan", "contact", "month", "campaign"]
    
    X = df[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cat_cols  = ["job", "marital", "education", "contact", "month"]
    bool_cols = ["housing", "loan"]
    num_cols  = ["age", "balance", "campaign"]

    preprocess_lr = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
            ("boo", "passthrough", bool_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    model = Pipeline(steps=[
        ("prep", preprocess_lr),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE))
    ])

   

    model.fit(X_train, y_train)
    pipe = model
    clf = pipe.named_steps["clf"]
    prep = pipe.named_steps["prep"]

    if isinstance(clf, LogisticRegression):
        names = prep.get_feature_names_out()  
        coefs = clf.coef_.ravel()      
        order = np.argsort(coefs)

        print("\nTop + (↑ YES):")
        for i in order[-10:][::-1]:
            print(f"{names[i]:<35} {coefs[i]: .3f}")

        print("\nTop - (↓ YES):")
        for i in order[:10]:
            print(f"{names[i]:<35} {coefs[i]: .3f}")
    else:
        print("\n[info]")

    proba = model.predict_proba(X_test)[:, 1]
    pred  = model.predict(X_test)
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"F1(macro): {f1_score(y_test, pred, average='macro'):.3f}")
    print(classification_report(y_test, pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model_pipeline.joblib")
    print("Saved -> models/best_model_pipeline.joblib")
    sample = pd.DataFrame([{
        "age": 30, "job": "technician", "marital": "single", "education": "tertiary",
        "balance": 1000.0, "housing": True, "loan": False, "contact": "cellular",
        "month": "may", "campaign": 1
    }])
    print("Sample pred:", model.predict(sample)[0])

if __name__ == "__main__":
    main()
