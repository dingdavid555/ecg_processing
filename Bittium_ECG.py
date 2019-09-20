# DAVID DING
# SEPTEMBER 16TH 2019           UPDATED SEPTEMBER 19TH 2019
# THIS FILE PROCESSES BITTIUM FILES


# ======================================== IMPORT STATEMENTS ========================================
import numpy as np
import pyedflib
import neurokit as nk
import matplotlib.pyplot as plt
import peakutils
from matplotlib import style
import joblib
import scipy.signal as signal
from scipy import integrate
import math
style.use("ggplot")


# ======================================== CLASSES AND OBJECTS ========================================
# Bittium_ECG stores all relevant files and methods to parse EDF file generated from the Bittium device
class BittiumECG:

    # Initialization Function
    def __init__(self, path, start, n_samples):
        # path (str) is the absolute file path of the EDF File
        # EDFFILE (pyedflib.EdfReader) stores the EDF class
        # WINDOW_LENGTH (int) stores the number of samples to store in each window
        # THRESHOLD (int) Arbitrary (temporarily) value for peak detection filtering
        # n_samples (int) Number of samples to get from the EDF file
        # ecg_raw (np.ndarray) Raw data stored in a 1-dimensional numpy array
        # ecg_filtered (np.ndarray) Filtered data after passing a bandpass filter on ecg_raw
        # ecg_preprocess (np.ndarray) Data used for binary peak detection, to be passed through peakutils
        # per_window (list) stores the expected value of each window of samples used for peak detection (WIP)
        # peak_indices (np.ndarray) returned array from the peakutils.indexes function
        # peak_window grabs the 150 Sample array from before and after the peak
        # nyq, freq, low, high (float) are all variables used for the bandpass_filter function
        self.path = path
        self.EDFFILE = pyedflib.EdfReader(path)
        self.WINDOW_LENGTH = 1250
        self.THRESHOLD = 25
        if n_samples == 0:
            self.ecg_raw = self.EDFFILE.readSignal(0, start=start)
        else:
            self.ecg_raw = self.EDFFILE.readSignal(0, start=0, n=n_samples)
        self.ecg_filtered = None
        self.ecg_gradients = None
        self.ecg_preprocess = None
        self.per_window = None
        self.peak_indices = None
        self.peak_window = []
        self.nyq = None
        self.freq = None
        self.low = None
        self.high = None

    # bandpass_filter applies a butterworth-bandpass filter onto the raw data to account for high-frequency noise and
    # baseline wander
    def bandpass_filter(self, freq, low, high, btype):
        self.freq = freq
        self.nyq = 0.5 * self.freq
        self.low = low/self.nyq
        self.high = high/self.nyq
        b, a = signal.butter(3, [self.low, self.high], btype)
        self.ecg_filtered = signal.lfilter(b, a, self.ecg_raw)

    # differentiate applies the numpy first differences function on the data, used for peak detection
    def differentiate(self):
        self.ecg_gradients = np.ediff1d(self.ecg_filtered)
        for i in range(len(self.ecg_gradients)):
            self.ecg_gradients[i] = math.fabs(self.ecg_gradients[i])

    # window finds the expected value of the ECG frequency, used for both outlier detection and peak detection
    # TODO: Finish this method
    def window(self):
        # Using method of Expected Value to find peaks
        self.per_window = []
        for i in range(0, len(self.ecg_filtered), self.WINDOW_LENGTH):
            self.per_window.append(self.ecg_filtered[i:i+self.WINDOW_LENGTH])

    # get_peaks grabs the peaks using the peakutils method, creates a graphable array (ecg_preprocess) to compare
    # the peak values and the filtered ecg with pyplot
    def get_peaks(self):
        self.ecg_preprocess = [0 for j in range(len(self.ecg_gradients))]
        for i in range(len(self.ecg_gradients)):
            if self.ecg_gradients[i] >= self.THRESHOLD:
                self.ecg_preprocess[i] = 50
            else:
                self.ecg_preprocess[i] = 0

        self.ecg_preprocess = np.asarray(self.ecg_preprocess)
        self.peak_indices = peakutils.indexes(self.ecg_preprocess, thres=0.3, min_dist=50)

    # get_peak_windows grabs samples from -50 to +100 of the peak frequency, excludes the first and last two samples in
    # case the file ends before the indices
    # TODO: Edge detection, graph only the segment that exists within the  filtered data file
    def get_peak_windows(self):
        for peak in self.peak_indices[1:-2]:
            self.peak_window.append(self.ecg_filtered[peak-50:peak+100].tolist())
