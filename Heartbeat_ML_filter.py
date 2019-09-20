# DAVID DING
# September 19th 2019
# This file returns the rolling heart rate and visual pyplots of health metrics
# TODO: FINISH THIS FILE


# ========================================= IMPORTS =========================================
import tensorflow
import keras
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle
from ECG_HR import *
from Bittium_ECG import *

# ========================================= VARIABLE AND CONSTANTS =========================================
# TIMEFRAME (const) (int) stores the length of time in Samples that each window is analyzed for heart beats for (BPM)
# file_name (str) stores the filename
# bio (BittiumECG) class instance variable
# model (sklearn.neighbors.KNeighborsClassifier) model used to detect heartbeats
# count stores the number of iteration that the current index is stored at
# BPMs is the array that stores all the BPMs of a certain input file
# BPM stores the current measured/observed BPM
# predictions (np.ndarray) used as output for the model.predict function for the input data
# ALTERNATE FILE FOR DEBUGGING: "/Volumes/nimbal$/Data/ReMiNDD/Raw data/Bittium/OND06_SBH_1039_01_SE01_GABL_BF_02.EDF"
TIMEFRAME = 1250
start = input("Please enter where to start: ")
n_samples = input("Please enter the number of samples: ")
file_name = "/Volumes/nimbal$/OBI/ONDRI@Home/Group A - In-Lab Trials/Group A - In-Lab Collections/Individual Participant Data/OF_1/Heart Rate Files/EDF Files/OF_Bittium_HR.EDF"
bio = BittiumECG(file_name, start, n_samples)
model = pickle.load(open("pickled_models/model3.pickle", "rb"))
count = 0
BPMs = []
BPM = 0


# ======================================== PREPROCESSING ========================================
bio.bandpass_filter(250, 3, 45, "bandpass")
bio.differentiate()
bio.get_peaks()
bio.get_peak_windows()

predictions = model.predict(np.asarray(bio.peak_window))


# ======================================== PARSING ========================================
# Removing all false positives
for i in range(len(predictions)):
    if predictions[i] == 0:
        bio.peak_indices.remove(bio.peak_indices[i])

# TODO: Finish this
# Plots the filtered data, with the false peaks removed
plt.figure(figsize=[14, 7])
plt.plot(bio.ecg_filtered)
arr = [0 for i in range(len(bio.ecg_filtered))]
for i in bio.peak_indices:
    arr[i] = 50
plt.plot(arr)

# BPM Calculations
for i in range(len(bio.ecg_filtered)):
    if i in bio.peak_indices:
        BPM += 1
    count += 1

    if count == TIMEFRAME:
        count = 0
        BPMs.append(BPM*12)
        BPM = 0

# Outputs
print(BPMs)
plt.plot([TIMEFRAME * i for i in range(len(BPMs))], BPMs)
plt.show()


