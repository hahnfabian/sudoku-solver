import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(file, num_puzzles = None):
    data = pd.read_csv(file)
    
    if num_puzzles is not None:
        data = data.head(num_puzzles)
    
    feature_raw = data['puzzle']
    label_raw = data['solution']

    feature = []
    label = []

    # converts the sudoku puzzle into a 9x9x1 array for the CNN
    for i in feature_raw:
        x = np.array([int(j) for j in i]).reshape((9,9,1))
        feature.append(x)
    
    feature = np.array(feature)
    
    # normalize and center around 0
    feature = feature/9 
    feature -= 0.5 
    
    for i in label_raw:
        x = np.array([int(j) for j in i]).reshape((81,1)) - 1
        label.append(x)   
    
    label = np.array(label)
    
    del(feature_raw)
    del(label_raw)    

    # splits the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test