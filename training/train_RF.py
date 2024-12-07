from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_RF(RUN_HYPERPARAMETER_SEARCH_RT, X_train, X_train_raw, y_train):

    param_grid = {
        'n_estimators': [200, 800, 1200],
        'max_depth': [2, 6, 15],
        'min_samples_split': [2, 4, 6, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if RUN_HYPERPARAMETER_SEARCH_RT:
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='neg_log_loss'
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters for model (X_train): ", grid_search.best_params_)
        rf_best_params = grid_search.best_params_
    else:
        rf_best_params = {
            'n_estimators': 1200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt'
        }

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **rf_best_params
    )
    model.fit(X_train, y_train)


    model_raw = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **rf_best_params
    )
    model_raw.fit(X_train_raw, y_train)

    return model, model_raw
