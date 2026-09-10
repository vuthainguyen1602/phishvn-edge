"""Build a deterministic synthetic model for exercising export and inference."""
from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    X, y = make_classification(n_samples=256, n_features=9, n_informative=5, random_state=42)
    model = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42).fit(X, y)
    path = Path("models/demo.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved {path}; 9 input features; synthetic demo only, not a phishing detector.")
