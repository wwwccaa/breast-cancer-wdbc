import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# 标准化必须放进 pipeline，否则会数据泄漏
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print("AUC per fold:", np.round(auc, 3))
print("AUC mean±std:", round(auc.mean(), 3), "±", round(auc.std(), 3))
print("ACC mean±std:", round(acc.mean(), 3), "±", round(acc.std(), 3))