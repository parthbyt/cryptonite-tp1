# Module 5: Sequnce Models

This module focused on creating models for sequential data like text and weather. We learned structures like RNNs, GRUs, and LSTMs.

## Jupyter Notebooks

### Weather Forecasting

This notebook revolves around predicting manipals weather using a univariate and a multivariate model.

| Model | Inputs | Temp MAE (°C) | Temp RMSE (°C) | Precip MAE (mm) | Precip RMSE (mm)
|----|----|----|----|----|----|
| Univariate RNN | Temperature | 0.39 | 0.51 | 6.07 | 11.01 |
| Multivariate GRU | All available variables | 0.37 | 0.49 | 5.73 | 10.61|

### Office script generator

Given a sample script of the show Office, we trained an character-level LSTM to generate a sequence based on some seed text.

**Train Perplexity:** 3.12
**Validation Preplexity:** 3.27 

We tested this model by generating a sequence of texts based on the same seed text at three different temperatures. 

**Best Temperature:** 0.7

### Climate Analysis EDA

This notebook contains exploratory data analysis on Manipal's Weather. Through this we could conclude - 
 - Manipal has two main seasons of summer and monsoon
 - Climate change has caused temperature to rise by **1.0** degree celcius over 15 years.
 - Rainfall has become less and less predictable.

## Reports

### Architecture of RNNs
This report contains an in detial intuitive explaination of the working of RNNs and LSTMs.

### Climate Analysis Summary

This report contains a summary of results and key finding of the EDA on Manipal's weather.