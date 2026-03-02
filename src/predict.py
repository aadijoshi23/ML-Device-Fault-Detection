import pandas as pd
import joblib

model = joblib.load("outputs/model.pkl")

test = pd.read_csv("data/TEST.csv")

ids = test["ID"]
X_test = test.drop(columns=["ID"])

pred = model.predict(X_test)

submission = pd.DataFrame({
    "ID": ids,
    "Class": pred
})

submission.to_csv("outputs/FINAL.csv", index=False)

print("Submission file created: outputs/FINAL.csv")