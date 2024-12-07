from xgboost import XGBClassifier


def train_GBDT(X_train, X_test, X_train_raw, X_test_raw, y_train, y_test):

    model = XGBClassifier(
        n_estimators=1440,
        max_depth=6,
        learning_rate=0.01,
        gamma=0.2871,
        min_child_weight=31,
        subsample=0.55,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        reg_alpha=0.006864,
        reg_lambda=2.057,
        max_delta_step=1,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    model_raw = XGBClassifier(
        n_estimators=1440,
        max_depth=6,
        learning_rate=0.01,
        gamma=0.2871,
        min_child_weight=31,
        subsample=0.55,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        reg_alpha=0.006864,
        reg_lambda=2.057,
        max_delta_step=1,
        objective='multi:softprob',
        eval_metric='mlogloss'
    )
    model_raw.fit(X_train_raw, y_train, eval_set=[(X_test_raw, y_test)], verbose=False)

    return model, model_raw
