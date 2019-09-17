import tensorflow
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Input
import sklearn

import numpy as np
import pandas as pd
import pyedflib
import neurokit as nk
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from datetime import *
import joblib
import scipy.signal as signal
style.use("ggplot")


class Bittium_ECG:
    path: str
    EDFFile: pyedflib.EdfReader
    ecg_raw: np.ndarray
    ecg_filtered: np.ndarray
    b: np.ndarray
    a: np.ndarray
    ecg_differentiated = np.ndarray
    ecg_gradients: np.ndarray

    def __init__(self, path):
        self.path = path
        self.EDFFILE = pyedflib.EdfReader(path)
        self.ecg_raw = self.EDFFILE.readSignal(0)   # TODO: CHANGE READSIGNAL TO READ MORE THAN THE FIRST N ELEMENTS
        self.ecg_filtered = None
        self.nyq = None
        self.freq = None
        self.low = None
        self.high = None

    def bandpass_filter(self, freq, low, high, btype):
        self.freq = freq
        self.nyq = 0.5 * self.freq
        self.low = low/self.nyq
        self.high = high/self.nyq
        b, a = signal.butter(3, [self.low, self.high], btype)
        self.ecg_filtered = signal.lfilter(b, a, self.ecg_raw)

    def differentiate(self):
        self.ecg_gradient = np.gradient(self.ecg_filtered)

    def activation_indices(self):
        return 0


input_file = r"O:\Data\ReMiNDD\Raw data\Bittium\OND06_SBH_1039_01_SE01_GABL_BF_02.EDF"

bio = Bittium_ECG(input_file)
bio.bandpass_filter(250, 3, 18, "bandpass")
bio.differentiate()

plt.figure(figsize=(18, 9))
# plt.plot(bio.ecg_raw)
plt.plot(bio.ecg_filtered)
plt.plot(bio.ecg_gradient)
plt.show()




