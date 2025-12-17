In this module, our primary focus was on multilayered perceptrons MLPs.
The two models I built were - 

## MLP From Scratch

Dataset - **Wine Quality Dataset**

I built a simple MLP from scratch, including all the functions. Using them, I constructed a neural network consisting of two hidden layers.

The goal was to predict Wine quality, and I decided to go with MLP classification over Regression due to slightly better results. 

I achieved an average accuracy of 54%, but with relatively low precision and recall. The model was highly biased towards average qualities.

## MLP using PyTorch

Dataset - **Adult Income Dataset**

I used the PyTorch library to build an MLP to perform binary classification of incomes on the dataset. The provided dataset actually required quite significant data preprocessing compared to the past. 

The goal was to predict whether the income earned was greater than 50K or not. I opted for dropout as a regularisation method.

I achieved a high accuracy of 85% with good precision and recall, although the model tends to choose lower incomes in disproportionately more cases.

## Reports

Finally, I have prepared reports on - 

- Backpropagation and Gradient-Based Optimisation
- Regularization in MLPs
- Weight Initialization and Training Stability