#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# config.py

# hyperparameters
batch_size = 16
validation_split = 0.1
learning_rate = 0.0005
epochs = 10000

lstm_units = 64
gru_units = 32
dropout_rate = 0.5
l2_reg = 0.02

optimizer_type = 'Adam'
beta_1 = 0.9
beta_2 = 0.999

noise_level = 0.1

# cross validation
n_splits = 10
n_repeats = 5
random_seed = 42

# file path
data_file_path_2022 = "/Users/sabai/Desktop/data/static/df1-2022.csv"
data_file_path_2019 = "/Users/sabai/Desktop/data/static/df1-2019.csv"

target_variable = 'aadb'

save_checkpoints = True
checkpoint_dir = './checkpoints'
verbose_level = 0

show_plots = True
plot_dir = './plots'

use_batch_norm = True
activation_function = 'relu'

