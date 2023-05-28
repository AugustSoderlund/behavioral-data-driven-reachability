import matplotlib.pyplot as plt
import numpy as np
from . import map

COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange"]

def visualize_class(map: map.SinD_map, _class: int, data: np.ndarray, labels: np.ndarray, input_len: int = 30):
    """ Visualize trajectories from a specific class

    Parameters:
    -----------
    map : SinD_map
        The map class for the dataset
    _class : int
        Specific class for the trajectory classification
    data : np.ndarray
        Data for the dataset
    labels : np.ndarray
        Labels for the data
    input_len : int
        Length of input for the data
    """
    ax, _ = map.plot_areas()
    _ids = np.where(labels == _class)
    _data = data[_ids]
    for _d in _data:
        x, y = _d[0:input_len], _d[input_len:2*input_len]
        ax.plot(x, y, color="r", linewidth=0.5)
    plt.show()

def visualize_all_classes(map: map.SinD_map, num_classes: int, data: np.ndarray, labels: np.ndarray, input_len: int = 30):
    """ Visualize the class for each trajectory

    Parameters:
    -----------
    map : SinD_map
        The map class for the dataset
    num_classes : int
        The number of classes for the classification
    data : np.ndarray
        Data for the dataset
    labels : np.ndarray
        Labels for the data
    input_len : int
        Length of input for the data
    """
    ax = map.plot_areas()
    for _class in range(num_classes):
        _ids = np.where(labels == _class)
        _data = data[_ids]
        for _d in _data:
            x, y = _d[0:input_len], _d[input_len:2*input_len]
            ax.plot(x, y, color=COLORS[_class], linewidth=1)
    plt.show()
