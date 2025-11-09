from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

def smote_balance(X_train, y_train, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))

def describe_distribution(y, label=""):
    c = Counter(y)
    total = sum(c.values())
    return {k: f"{v} ({v/total:.2%})" for k, v in c.items()}
