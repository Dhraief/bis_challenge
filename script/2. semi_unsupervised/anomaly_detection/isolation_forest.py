from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score

def run_isolation_forest(X, y, contamination=0.05):
    """
    Use Isolation Forest for anomaly detection with `y`.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(X)

    # Convert {-1, 1} to {0, 1} for evaluation
    preds = (preds == 1).astype(int)
    y = (y == 1).astype(int)

    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    print(f"Isolation Forest Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    return preds
