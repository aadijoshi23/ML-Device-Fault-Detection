import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

train = pd.read_csv("data/TRAIN.csv")

X = train.drop(columns=["Class"])
y = train["Class"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
acc = accuracy_score(y_val, val_pred)
print("Validation Accuracy:", acc)

joblib.dump(model, "outputs/model.pkl")