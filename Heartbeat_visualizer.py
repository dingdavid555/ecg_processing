# DAVID DING
# SEPTEMBER 18TH 2019
# THIS FILE IS USED TO GENERATE WINDOWS FOR ML ALGORITHMS TO DETECT HEARTBEATS GIVEN .EDF FILES FOR PROCESSING


# ======================================== IMPORTS ========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bittium_ECG import *
from matplotlib import style


# ======================================== VARIABLE DECLARATION ========================================
# input_file (str) stores the absolute path for the EDF file to be read in
# MESSY_DATA_PATH (const) (str) Messy edf data, used for debugging purposes
# CLEAN_DATA_PATH (const) (str) Clean edf data, used for debugging purposes
# edf_file (BittiumECG Object)
# cols (list) names of the headers for the CSV file and dataframe
# df (pandas.DataFrame) Pandas DataFrame that stores all the values
# curr_arr (list) list of arrays that hold potential heartbeats to be validated
# counter counts how many graphs have been reviewed so far
MESSY_DATA_PATH = "/Volumes/nimbal$/Data/ReMiNDD/Raw data/Bittium/OND06_SBH_1039_01_SE01_GABL_BF_02.EDF"
CLEAN_DATA_PATH = "/Volumes/nimbal$/OBI/ONDRI@Home/Group A - In-Lab Trials/Group A - In-Lab Collections/Individual Participant Data/OF_1/Heart Rate Files/EDF Files/OF_Bittium_HR.EDF"
input_file = CLEAN_DATA_PATH
start_index = input("Please enter the starting sample index: \n")
num_of_samples= input("Please enter how many samples to extract: \n")
edf_file = BittiumECG(input_file, 10000, 110000)
cols = [i for i in range(150)] + ["IsHB"]
df = pd.DataFrame(columns=cols)
curr_arr = []
counter = 0

# Preprocessing the EDF file before we can see it graphically
edf_file.bandpass_filter(250, 3, 18, "bandpass")
edf_file.differentiate()
edf_file.get_peaks()

# Sets up the matplotlib.pyplot variables
style.use("ggplot")
fig = plt.figure()
ax = fig.add_subplot(111)
li, = ax.plot(edf_file.ecg_filtered[edf_file.peak_indices[1]-50:edf_file.peak_indices[1]+101])
fig.canvas.draw()
plt.show(block=False)

# Iterates through the peak_indices array to generate graphs
for peak in edf_file.peak_indices[1:]:  # Starting at index 1 since the first peak could be < 50 Samples from start
    # Get data to graph
    potential_heart_beat = edf_file.ecg_filtered[peak - 50:peak + 101]

    # ======================================== SOME ECG SIGNALS ARE FLIPPED, UNCOMMENT THIS TO UNFLIP THEM
    '''for i in range(len(potential_heart_beat)):
        potential_heart_beat[i] = potential_heart_beat[i] * -1'''

    # Update scales and graphs
    li.set_ydata(potential_heart_beat)
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.draw()

    # Gets user input to see if the graph contains a valid heartbeat or not
    isHR = input("1 or 0: \n")
    while isHR != "1" and isHR != "0":
        isHR = input("Invalid input, try again")

    # Stores the the boolean value of whether or not this is a heartbeat as the last entry of the array
    potential_heart_beat[150] = int(isHR)
    curr_arr.append(potential_heart_beat)
    counter += 1
    print("%i/%i" % (counter, len(edf_file.peak_indices)//5))

# Push data out to CSV for easier reading
df = pd.DataFrame(curr_arr, columns=cols)
df.to_csv("TESTS/TEST5.csv")
