# bike-project
Enhancing Bicyclist Volume Estimation with LSTM-GRU-Attention Model

This repository contains code for building and evaluating deep learning models that use LSTM, GRU, and Attention mechanisms for network-level bicycle volume estimation. The models use a combination of Strava and static features to predict bicycle volumes.

config.py: This file contains configuration settings for models, including hyperparameters such as number of epochs, batch size, learning rate, and optimizer settings.

features.py: Defines the different feature sets used in the model, including the Strava feature list, the static feature list, and the static + Strava feature list.

LSTM+GRU+ATTENTION Model.py: This model combines LSTM, GRU, and Attention layers. This model is designed to achieve optimal performance by integrating these powerful deep learning techniques.

LSTM.py: Contains an implementation of the LSTM-based model used to predict bicycle volume.

LSTM+ATTENTION.py: implements an LSTM model with an attention mechanism to improve prediction accuracy.

GRU.py: Contains the implementation of the GRU-based model used to predict bicycle volume.

LSTM+GRU.py: combines both LSTM and GRU layers into a single model to take advantage of the strengths of both architectures.
