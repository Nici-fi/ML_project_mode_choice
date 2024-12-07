import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def data_preprocessing(df_train):

    categorical_features = ['purpose', 'fueltype', 'faretype']
    df_train_categorical = df_train[categorical_features]
    df_train_preprocessed = pd.get_dummies(df_train_categorical, prefix=categorical_features)

    continuous_features = ['age', 'distance', 'cost_transit', 'bus_scale', 'dur_walking', 'dur_cycling', 'dur_pt_total',
                              'dur_pt_access', 'dur_pt_rail', 'dur_pt_bus', 'dur_pt_int_total', 'dur_pt_int_waiting',
                              'dur_pt_int_walking', 'pt_n_interchanges', 'dur_driving', 'cost_driving_total',
                              'cost_driving_fuel', 'cost_driving_con_charge', 'driving_traffic_percent']
    df_train_preprocessed[continuous_features] = df_train[continuous_features]

    temporal_features = ['travel_month', 'start_time_linear', 'day_of_week']
    df_train_preprocessed[temporal_features] = df_train[temporal_features]

    # Include travel_year for splitting the data set into training and test data
    df_train_preprocessed['travel_year'] = df_train['travel_year']

    df_train_preprocessed['target'] = df_train['travel_mode']

    # Temporal Preprocessing - Sine and Cosine transformations
    for feature in temporal_features:
        df_train_preprocessed[feature + '_sin'] = np.sin(2 * np.pi * df_train[feature] / df_train[feature].max())
        df_train_preprocessed[feature + '_cos'] = np.cos(2 * np.pi * df_train[feature] / df_train[feature].max())

    # Create a new column to define the train/test split
    df_train_preprocessed['split'] = 'train'
    df_train_preprocessed.loc[
        (df_train_preprocessed['travel_year'] == 2014) & (df_train_preprocessed['travel_month'] >= 4), 'split'
    ] = 'test'
    df_train_preprocessed.loc[
        df_train_preprocessed['travel_year'] > 2014, 'split'
    ] = 'test'

    # Split into training and testing datasets
    df_train_data = df_train_preprocessed[df_train_preprocessed['split'] == 'train']
    df_test_data = df_train_preprocessed[df_train_preprocessed['split'] == 'test']

    # Drop the 'split' column as it's not needed for the model
    df_train_data = df_train_data.drop(columns=['split','travel_year'])
    df_test_data = df_test_data.drop(columns=['split','travel_year'])

    # Separate features and target
    X_train = df_train_data.drop(columns=['target'])
    X_test = df_test_data.drop(columns=['target'])

    # Encode the target column
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train_data['target'])
    y_test = label_encoder.transform(df_test_data['target'])

    # Create a subset of the full dataset that only contains the raw data
    columns_to_delete = [
        'dur_walking', 'dur_cycling', 'dur_pt_rail', 'dur_pt_bus', 'dur_pt_access',
        'dur_pt_int_total', 'dur_pt_total', 'cost_transit', 'pt_n_interchanges',
        'dur_driving', 'cost_driving_total', 'cost_driving_con_charge', 'driving_traffic_percent'
    ]

    X_train_raw = X_train.drop(columns=columns_to_delete)
    X_test_raw = X_test.drop(columns=columns_to_delete)
    return X_train, X_test, X_train_raw, X_test_raw, y_train, y_test
