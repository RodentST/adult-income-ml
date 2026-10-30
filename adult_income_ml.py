import random;
import numpy as np;
from sklearn.datasets import fetch_openml;

FixedRandom = 42;
random.seed(FixedRandom);
np.random.seed(FixedRandom);


def load():
        dataset = fetch_openml(name='adult', version=2, as_frame=True);
        data = dataset.data;
        target = dataset.target;
        return data, target;


X, y = load();