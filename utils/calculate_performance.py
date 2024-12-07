import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def calculate_performance(model_name, model, model_raw, X_test, X_test_raw, y_test):

    # GBDT: Performance Metrics
    y_pred = model.predict(X_test)
    y_pred_raw = model_raw.predict(X_test_raw)
    y_prob = model.predict_proba(X_test)  # Returns probabilities for each class
    y_prob_raw = model_raw.predict_proba(X_test_raw)  # Same for raw data

    # Negative Log-Likelihood (NLL)
    nll = log_loss(y_test, y_prob)
    nll_raw = log_loss(y_test, y_prob_raw)

    # Expected Simulation Error (ESE)
    ese = 1 - np.mean([probs[true_class] for probs, true_class in zip(y_prob, y_test)])
    ese_raw = 1 - np.mean([probs[true_class] for probs, true_class in zip(y_prob_raw, y_test)])

    # Multiclass Classification Error (CE)
    ce = 1 - accuracy_score(y_test, y_pred)
    ce_raw = 1 - accuracy_score(y_test, y_pred_raw)

    print("Negative Log-Likelihood - " + model_name + " (NLL): " + f"{nll}")
    print(f"Expected Simulation Error - " + model_name + " (ESE): " + f"{ese}")
    print(f"Multiclass Classification Error - " + model_name + " (CE): " + f"{ce}")
    print(f"________________________________________________________________________")
    print(f"Negative Log-Likelihood raw - " + model_name + " (NLL): " + f"{nll_raw}")
    print(f"Expected Simulation Error raw - " + model_name + " (ESE): " + f"{ese_raw}")
    print(f"Multiclass Classification Error raw - " + model_name + " (CE): " + f"{ce_raw}")
    return nll, nll_raw, ese, ese_raw, ce, ce_raw