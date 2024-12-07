import pandas as pd
from utils.analysis import analysis
from training.train_GBDT import train_GBDT
from training.train_RF import train_RF
from training.train_MLP import train_MLP
from utils.calculate_performance import calculate_performance
from utils.data_preprocessing import data_preprocessing
import os


# Set flags
RUN_ANALYSIS= False
RUN_HYPERPARAMETER_SEARCH_RT= False
RUN_HYPERPARAMETER_SEARCH_MLP = False

# Load data set
current_path = os.getcwd()
data_file_path = current_path + '/data/londondataset.csv'
df_train = pd.read_csv(data_file_path)

if RUN_ANALYSIS:
    analysis(df_train)

# Preprocess data
[X_train, X_test, X_train_raw, X_test_raw, y_train, y_test] = data_preprocessing(df_train)


# Train GBDT
[gbdt_model, gbdt_model_raw] = train_GBDT(X_train, X_test, X_train_raw, X_test_raw, y_train, y_test)

# GBDT: Performance Metrics
[nll_gbdt, nll_gbdt_raw, ese_gbdt, ese_gbdt_raw, ce_gbdt, ce_gbdt_raw] = calculate_performance("GBDT", gbdt_model, gbdt_model_raw, X_test, X_test_raw, y_test)


# Train RF
[rf_model, rf_model_raw] = train_RF(RUN_HYPERPARAMETER_SEARCH_RT, X_train, X_train_raw, y_train)

# RF: Performance Metrics
[nll_rf, nll_rf_raw, ese_rf, ese_rf_raw, ce_rf, ce_rf_raw] = calculate_performance("RF", rf_model, rf_model_raw, X_test, X_test_raw, y_test)


# Train MLP
[mlp_model, mlp_model_raw] = train_MLP(RUN_HYPERPARAMETER_SEARCH_MLP, X_train, X_test, X_train_raw, X_test_raw, y_train, y_test)

# MLP: Performance Metrics
[nll_mlp, nll_mlp_raw, ese_mlp, ese_mlp_raw, ce_mlp, ce_mlp_raw] = calculate_performance("MLP", mlp_model, mlp_model_raw, X_test, X_test_raw, y_test)


results = {
    'Model': ['Random Forest', 'Random Forest raw', 'MLP', 'MLP raw', 'GBDT', 'GBDT raw'],
    'Negative Log-Likelihood (NLL)': [nll_rf, nll_rf_raw, nll_mlp, nll_mlp_raw, nll_gbdt, nll_gbdt_raw],
    'Expected Simulation Error (ESE)': [ese_rf, ese_rf_raw, ese_mlp, ese_mlp_raw, ese_gbdt, ese_gbdt_raw],
    'Multiclass Classification Error (CE)': [ce_rf, ce_rf_raw, ce_mlp, ce_mlp_raw, ce_gbdt, ce_gbdt_raw]
}

# Create a data frame from the result dictionary
df_comparison = pd.DataFrame(results)

df_comparison.to_csv('results/results.csv', index=False)
pd.set_option('display.max_columns', None)
print(df_comparison)