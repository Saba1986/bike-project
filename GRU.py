#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LSTM + ATTENTION Model

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Layer, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import time
from features import strava_static_features, strava_features, static_features
import config

# GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# build the model
def build_model(input_shape):
    model = Sequential([
        GRU(config.gru_units, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(config.l2_reg)),
        Dropout(config.dropout_rate),
        BatchNormalization() if config.use_batch_norm else tf.keras.layers.Layer(),
        GRU(config.gru_units, return_sequences=True, 
            kernel_regularizer=l2(config.l2_reg)),
        Dropout(config.dropout_rate),
        Dense(1)
    ])
    
    if config.optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=config.learning_rate, 
                         beta_1=config.beta_1, 
                         beta_2=config.beta_2)
    elif config.optimizer_type == 'SGD':
        optimizer = SGD(learning_rate=config.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# evaluate the model
def evaluate_model(X, y):
    rkf = RepeatedKFold(n_splits=config.n_splits, 
                        n_repeats=config.n_repeats, 
                        random_state=config.random_seed)
    rmse_scores = []
    prmse_scores = []
    mae_scores = []
    mape_scores = []
    mpe_scores = []
    r2_scores = []
    
    start_time = time.time()

    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 
                                                 1, 
                                                 X_train_scaled.shape[1]))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 
                                               1, 
                                               X_test_scaled.shape[1]))

        model = build_model((X_train_scaled.shape[1], 
                             X_train_scaled.shape[2]))
        history = model.fit(X_train_scaled, 
                            y_train, 
                            epochs=config.epochs,  
                            batch_size=config.batch_size,  
                            validation_split=config.validation_split, 
                            verbose=config.verbose_level)

        y_pred = model.predict(X_test_scaled).flatten()
        y_test = np.nan_to_num(y_test)
        y_pred = np.nan_to_num(y_pred)

        # metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        prmse = (rmse / np.mean(y_test)) * 100
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mpe = np.mean((y_test - y_pred) / y_test) * 100
        r2 = r2_score(y_test, y_pred)

        rmse_scores.append(rmse)
        prmse_scores.append(prmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        mpe_scores.append(mpe)
        r2_scores.append(r2)

    end_time = time.time()
    
    results = {
        "Mean RMSE": np.mean(rmse_scores),
        "SE RMSE": np.std(rmse_scores) / np.sqrt(len(rmse_scores)),
        "Mean PRMSE": np.mean(prmse_scores),
        "SE PRMSE": np.std(prmse_scores) / np.sqrt(len(prmse_scores)),
        "Mean MAE": np.mean(mae_scores),
        "SE MAE": np.std(mae_scores) / np.sqrt(len(mae_scores)),
        "Mean MAPE": np.mean(mape_scores),
        "SE MAPE": np.std(mape_scores) / np.sqrt(len(mape_scores)),
        "Mean MPE": np.mean(mpe_scores),
        "SE MPE": np.std(mpe_scores) / np.sqrt(len(mpe_scores)),
        "Mean R²": np.mean(r2_scores),
        "SE R²": np.std(r2_scores) / np.sqrt(len(r2_scores)),
        "Training Time": end_time - start_time
    }
    
    return results, history

for data_file_path in [config.data_file_path_2022, 
                       config.data_file_path_2019]:
    print(f"Processing dataset: {data_file_path}")
    
    data = pd.read_csv(data_file_path)
    data = data.select_dtypes(exclude='object')
    data.fillna(method='ffill', inplace=True)

    feature_sets = {
        'Strava + Static Features': strava_static_features,
        'Strava Features': strava_features,
        'Static Features': static_features
    }

    y = data[config.target_variable]
    y.fillna(y.median(), inplace=True)

    def add_noise(X, noise_level=config.noise_level):
        noise = np.random.normal(loc=0, 
                                 scale=noise_level, 
                                 size=X.shape)
        X_noisy = X + noise
        return X_noisy

    data['aadb_bin'] = pd.qcut(data[config.target_variable], 
                               q=[0, 0.333, 0.666, 1], 
                               labels=['1', '2', '3'])

    bin_counts = data['aadb_bin'].value_counts()
    print(bin_counts)

    max_values = data.groupby('aadb_bin')[config.target_variable].max()

    print("\nMaxi for each bin:")
    print(max_values)

    total_entries = len(data)
    print("\nTotal in the whole dataset:")
    print(total_entries)

    # Evaluate model for each bin
    for feature_set_name, features in feature_sets.items():
        print(f"\nEvaluating model for {feature_set_name}...")
        for label in data['aadb_bin'].unique():
            bin_data = data[data['aadb_bin'] == label]
            X_bin = bin_data[features]
            y_bin = bin_data[config.target_variable]
            X_bin_augmented = np.vstack([X_bin, add_noise(X_bin)])
            y_bin_augmented = np.hstack([y_bin, y_bin])
            bin_results, history = evaluate_model(X_bin_augmented, y_bin_augmented)

            print(f"Results for {feature_set_name} - Bin {label}:")
            for metric, score in bin_results.items():
                print(f"{metric}: {score}")
            print()
            
            if config.show_plots:
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title(f'Training and Validation Loss for {feature_set_name} - Bin {label}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

    # Evaluate model for the whole data
    for feature_set_name, features in feature_sets.items():
        print(f"\nEvaluating model with {feature_set_name} on the whole data...")
        X_whole = data[features]
        X_whole_augmented = np.vstack([X_whole, add_noise(X_whole)])
        y_whole_augmented = np.hstack([y, y])
        whole_results, history = evaluate_model(X_whole_augmented, y_whole_augmented)

        print(f"Results for {feature_set_name} - Whole Data:")
        for metric, score in whole_results.items():
            print(f"{metric}: {score}")
        print()

        if config.show_plots:
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss for {feature_set_name} - Whole Data')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

