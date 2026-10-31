from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = load_breast_cancer(as_frame=True)
df = data.frame
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})

print("rows:", len(df))
print(df["diagnosis"].value_counts().to_string())

plt.figure(figsize=(9, 6))
for label, color in [("malignant", "red"), ("benign", "blue")]:
    part = df[df["diagnosis"] == label]
    plt.scatter(part["mean radius"], part["mean texture"], s=18, c=color, label=label, alpha=0.75)

plt.xlabel("mean radius")
plt.ylabel("mean texture")
plt.title("WDBC: radius vs texture")
plt.legend()
plt.tight_layout()
plt.savefig("cancer_plot.png", dpi=150)
print("saved: cancer_plot.png")