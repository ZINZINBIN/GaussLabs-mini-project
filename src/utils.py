import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def MSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = True)

def RMSE(gt : np.array, pt : np.array):
    return mean_squared_error(gt, pt, squared = False)

def MAE(gt: np.array, pt: np.array):
    return np.mean(np.abs((gt - pt)))

def compute_metrics(gt : Union[np.ndarray, List], pt : Union[np.ndarray, List], algorithm : Optional[str] = None, is_print : bool = True):
    
    mse = MSE(gt, pt)
    rmse = RMSE(gt, pt)
    mae = MAE(gt, pt)
    
    if is_print:
        if algorithm:
            print("# {}, mse : {:.3f}, rmse : {:.3f}, mae : {:.3f}".format(algorithm, mse, rmse, mae))
        else:
            print("# mse : {:.3f}, rmse : {:.3f}, mae : {:.3f}".format(mse, rmse, mae))
            
    return mse, rmse, mae

def split_data(df : pd.DataFrame, train_ratio : float, valid_ratio : float):
    total_len = len(df)
    train_len = int(total_len * train_ratio)
    valid_len = int(total_len * valid_ratio)
    
    df_train = df[0:train_len]
    df_valid = df[train_len:train_len + valid_len]
    df_test = df[train_len + valid_len:-1]
    
    return df_train, df_valid, df_test


def plot_learning_curve(train_loss, valid_loss, figsize : Tuple[int,int] = (6,4), save_dir : str = "./results/learning_curve.png"):
    x_epochs = range(1, len(train_loss) + 1)

    plt.figure(1, figsize=figsize, facecolor = 'white')
    plt.plot(x_epochs, train_loss, 'ro-', label = "train loss")
    plt.plot(x_epochs, valid_loss, 'bo-', label = "valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train and valid loss curve")
    plt.legend()
    plt.savefig(save_dir)