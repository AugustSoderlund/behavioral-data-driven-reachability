from sklearn.model_selection import train_test_split
from .input_state import *
from .data_reader import SinD
from typing import List
import numpy as np
import pickle
import os

ROOT = os.getcwd()


def process_data(data: np.ndarray, input_len: int):
    """Subtract the mean from (x,y) from data

    Parameters:
    -----------
    data_traj : np.ndarray
        The dataset to be processed
    input_len : int
        Length of each trajectory
    """
    _processed_data = []
    for _d in data:
        _x, _y = _d[0:input_len], _d[input_len : 2 * input_len]
        _x, _y = _x - np.mean(_x), _y - np.mean(_y)
        _processed_data.append(np.array([*_x, *_y, *_d[2 * input_len :]]))
    return _processed_data


def load_data(file: str = "sind.pkl") -> np.ndarray:
    """Load previously pickled data

    Parameters:
    -----------
    file : str (default = 'sind.pkl')
        File-name of the pickled file. Note that
        it has to be stored in the relative dir
        '/thesis_project/.datasets/'
    """
    _f = open(ROOT + "/resources/" + file, "rb")
    return pickle.load(_f)


def save_data(data: np.ndarray, file: str):
    """Save data into file

    Parameters:
    -----------
    data : np.ndarray
        The data that will be pickled and stored
    file : str
        Name of the file it will be stored into
        relative dir: '/thesis_project/.datasets/'
    """
    _f = open(ROOT + "/resources/" + file, "rb")
    pickle.dump(data, _f)


def split_data(
    data: np.ndarray, labels: np.ndarray, test_size: float = 0.2, shuffle: bool = True
) -> List[np.ndarray]:
    """Split the data into train- and test data
    (this function randomizes the order or the data)

    Parameters:
    -----------
    data : np.ndarray
        The dataset that will be splitted
    test_size : float (default = 0.2)
        Size of the test dataset
    """
    return train_test_split(
        data,
        labels,
        test_size=test_size,
        train_size=1 - test_size,
        shuffle=shuffle,
    )


def label_data(
    sind: SinD, data: np.ndarray, input_len: int = 90, save: str = None
) -> np.ndarray:
    """Function for getting the labels for a dataset

    Parameters:
    -----------
    sind : SinD
        The SinD-class which calls the function
    data : np.ndarray
        The data to be labeled
    input_len : int (default = 30)
        The length for each chunk in the dataset
    save : str (default = None)
        Name of the file if the labels will be stored
        (will save if not set to None)
    """
    if save:
        assert type(save) == str
        save_data(data, save)
    return sind.labels(data, input_len)


def structure_input_data(data: np.ndarray, labels: np.ndarray):
    """Drops random trajectories such that the data for each class
    is of the same length

    Parameters:
    -----------
    data : np.ndarray
        The chunks from the dataset
    labels: np.ndarray
        The true labels for the data
    """
    _d = {}.fromkeys(labels)
    [_d.update({i: []}) for i in _d.keys()]
    [_d[_l].append(data[i]) for i, _l in enumerate(labels)]
    _lens = [len(v) for v in _d.values()]
    _min_len = min(_lens)
    new_d = []
    new_l = []
    for _l, _v in _d.items():
        _v = np.array(_v)
        _ids = np.random.randint(0, len(_v), size=_min_len)
        new_d = [*new_d, *_v[_ids]]
        new_l = [*new_l, *[_l] * _min_len]
    return np.array(new_d), np.array(new_l)
