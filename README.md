# LSTM_autoencoder

Autoencoder for anomaly detection.

# Daily Female Births Dataset

This dataset describes the number of daily female births in California in 1959. The units are a count and there are 365 observations. The source of the dataset is credited to Newton (1988).

Below is a plot of the entire dataset.

<img src="img/Daily-Female-Births-Dataset.png" width="700"/>

Dataset has two columns: Date and Births.

<img src="img/dataset_columns.png" width="600"/>

# Data Split

We will split data into training and validation dataset to train our autoencoder.

And then we will represent the data as a list of tensors.

<img src="img/data_split.png" width="700"/>

# Abnormal data

To test the autoencoder we will generate fake abnormal data.

<img src="img/fake_data_generation.png" width="800"/>

Below is a plot of generated abnormal data.

<img src="img/abnormal_data.png" width="800"/>

# LSTM Autoencoder

Autoencoder consists of encoder and decoder.

In encoder we will have 2 LSTM layers with embedding dimension equal to 64.

<img src="img/encoder.png" width="800"/>

In decoder we will have 2 LSTM layers and 1 fully connected layer.

<img src="img/decoder.png" width="800"/>

# Training

We will train our model on 75 epochs.

Loss on last epoch: train loss 0.5739964389801026 val loss 0.7613655783913352

<img src="img/loss_plot.png" width="800"/>

# Prediction

We will try to make a prediction and calculate losses on normal data and abnormal generated data.

<img src="img/predict_func.png" width="800"/>

Plot of real data and Autoencoder prediction.

<img src="img/autoencoder_prediction.png" width="800"/>

Plot of losses distributions on training data.

<img src="img/losses_dist.png" width="800"/>

As we can see most of the errors are less than 0.9. So our threshold will be 0.9.

On train data correct normal predictions: 191/200 (95%).

# Prediction on anomaly data

Lets make prediction on anomaly data and calculate the losses.

<img src="img/anomaly_prediction.png" width="800"/>

Plot of losses distributions on anomaly data.

<img src="img/anomaly_loss_dist.png" width="800"/>

As we can see majority of observations have loss more than our threshold.

On abnormal data correct anomaly predictions: 136/200 (68%).
