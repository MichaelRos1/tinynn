import numpy as np

def mse(labels: np.array, predictions: np.array):
    return np.mean((labels-predictions)**2)

def mae(labels: np.array, predictions: np.array):
    return np.mean(np.abs(labels-predictions))

def bce(labels: np.array, predictions: np.array):
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)      #Prevent log(predictions)=inf/-inf 
    loss = labels * np.log(predictions) + (1-labels) * np.log(1-predictions)
    return -np.mean(loss)

def cce(labels: np.array, predictions: np.array):
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)      #Prevent log(predictions=inf/-inf
    loss = np.sum(labels * np.log(predictions), axis=1)
    return -np.mean(loss)
