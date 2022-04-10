import deepdish as dd
import numpy as np
import cv2

def load_model():
  model = dd.io.load('static/model/model.h5')
  return model

def loadImgData(image):
    data = cv2.resize(image, (32, 32))
    (b, g, r) = cv2.split(data)
    data = cv2.merge([r,g,b])
    data = data / 255.
    data = data.reshape((1, 32*32*3)).T
    return data

def predict(X):
    parameters = load_model()
    AL, cache = forward_propagation_regularization(X, parameters)
    predictions = (AL > 0.5)
    return predictions

def forward_propagation_regularization(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward_regularization(A, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward_regularization(A, parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
          
    return AL, caches

def linear_activation_forward_regularization(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward_regularization(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward_regularization(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def linear_forward_regularization(A, W, b):
    Z= np.dot(W, A) + b

    cache = (A, W, b)
    
    return Z, cache

def sigmoid(Z): 
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):  
    A = np.maximum(0.001,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache