import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def train_MLP(RUN_HYPERPARAMETER_SEARCH_MLP, X_train, X_test, X_train_raw, X_test_raw, y_train, y_test):
    param_grid = {
        'hidden_layer_sizes': [
            (100,),
            (50, 50),
            (100, 100),
            (50, 50, 50),
            (100, 75, 50),
            (100, 100, 100),
            (50, 50, 50, 50),
            (100, 75, 50, 25),

        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],  # L2
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000],
    }

    y_combined = np.concatenate([y_train, y_test])
    X_combined = np.concatenate([X_train, X_test])
    X_combined_raw = np.concatenate([X_train_raw, X_test_raw])
    validation_fraction = len(X_test) / len(X_combined)

    mlp_model = MLPClassifier(
        batch_size='auto',
        random_state=42,
        verbose=False,
        n_iter_no_change=35,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=validation_fraction,
    )

    if RUN_HYPERPARAMETER_SEARCH_MLP:
        grid_search = GridSearchCV(
            estimator=mlp_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_log_loss'
        )
        grid_search.fit(X_combined, y_combined)
        print("Best parameters found: ", grid_search.best_params_)
        mlp_best_params = grid_search.best_params_
    else:
        mlp_best_params = {
            'activation': 'relu',
            'alpha': 0.0001,
            'hidden_layer_sizes': (100, 75, 50, 25),
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'solver': 'adam'
        }

    model = MLPClassifier(
        batch_size='auto',
        random_state=42,
        verbose=False,
        n_iter_no_change=35,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=validation_fraction,
        **mlp_best_params
    )
    model.fit(X_combined, y_combined)

    model_raw = MLPClassifier(
        batch_size='auto',
        random_state=42,
        verbose=False,
        n_iter_no_change=35,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=validation_fraction,
        **mlp_best_params
    )
    model_raw.fit(X_combined_raw, y_combined)

    return model, model_raw


