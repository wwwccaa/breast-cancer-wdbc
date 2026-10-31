from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # 0 = malignant, 1 = benign
df = data.frame

print(df.groupby("target")["mean radius"].agg(["mean", "min", "max"]).to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000).fit(X_train_s, y_train)
pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]

print("accuracy:", round(accuracy_score(y_test, pred), 3))
print("auc:", round(roc_auc_score(y_test, proba), 3))
print("confusion matrix:")
print(confusion_matrix(y_test, pred))