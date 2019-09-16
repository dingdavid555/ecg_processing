# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ PACKAGE IMPORT ----------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import csv
from time import gmtime, strftime
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
import os
from itertools import zip_longest
import statistics as stats
from random import randint
import math
import pyedflib
import peakutils
import heartpy
from scipy.signal import resample

import tensorflow
import keras
import sklearn


# Datetime formatting for X-axis on plots
xfmt = mdates.DateFormatter("%a @ %H:%M:%S:%f"[0:-3])  # Date as "<day of week> @ HH:MM:SS"
locator = mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21], interval=1)  # Tick marks every 3 hours

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------ CLASS: EDF FILE ---------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class EdfFile(object):

    def __init__(self, file, crop, epoch_len):

        # Passes in arguments from call to class instance
        self.filename = file  # filename including full pathway
        self.file_id = self.filename.split(".", 1)[0]  # filename only, no extension

        self.epoch_len = epoch_len  # in seconds
        self.crop = crop  # seconds to crop at start and end of file, list

        self.raw_file = None  # object that stores EDF object

        self.signal_frequency = None  # sampling frequency, Hz
        self.starttime = None  # timestamp of collection start
        self.collection_duration = None  # in seconds
        self.collection_duration_days = None  # in days

        self.load_time = None  # time it takes to load file

        self.ecg_raw = None  # raw ECG data
        self.raw_timestamps = None  # timestamps for each data point

        self.epoch_timestamps = []  # timestamp for each epoch

        # Runs function
        self.load_edf()
        self.create_epochs()

    def load_edf(self):
        """Loads ECG channel from EDF file and calculates timestamp for each datapoint. Crops time off beginning
           and end of file."""

        t0_load = datetime.now()

        print("\n" + "----------------------------- Loading entire data file ----------------------------------")
        print("Loading {} ...".format(self.filename))

        # Reads in EDF file - ECG channel only
        self.raw_file = pyedflib.EdfReader(self.filename)

        # Reads in ECG channel only
        self.ecg_raw = np.zeros((1, self.raw_file.getNSamples()[0]))

        for i in np.arange(1):
            self.ecg_raw[i, :] = self.raw_file.readSignal(i)

        # Reshapes raw ECG signal to long format
        self.ecg_raw = self.ecg_raw.reshape(self.ecg_raw.shape[1], 1)

        # Reads in data from header
        self.signal_frequency = self.raw_file.getSampleFrequencies()[0]  # signal frequency
        self.starttime = self.raw_file.getStartdatetime()  # start timestamp

        # Crops start and end of file
        self.ecg_raw = self.ecg_raw[self.crop[0]*self.signal_frequency:
                                    len(self.ecg_raw) - self.crop[1]*self.signal_frequency]

        # Duration of collection in seconds and days, respectively
        self.collection_duration = self.raw_file.file_duration  # in seconds

        self.collection_duration_days = self.collection_duration/3600/24  # in hours

        # Creates a timestamp for every data point
        self.raw_timestamps = [(self.starttime + timedelta(seconds=i / self.signal_frequency))
                               for i in range(0, len(self.ecg_raw))]

        print("\n" + "Data loaded. File duration is {} hours.".format(round(self.collection_duration / 3600), 1))

        t1_load = datetime.now()

        self.load_time = round((t1_load - t0_load).seconds, 1)

        print("Data import time is {} seconds.".format(self.load_time))

    def create_epochs(self):
        """Creates a list of timestamps corresponding to starttime + epoch_len."""

        # Creates list of timestamps corresponding to starttime + epoch_len for each epoch
        for i in range(0, int(self.collection_duration / self.epoch_len)):
            timestamp = self.raw_timestamps[0] + timedelta(seconds=i * self.epoch_len)
            self.epoch_timestamps.append(timestamp)


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- PAIKRO SCRIPT ----------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class PaikroAlgorithm(object):

    def __init__(self, filter_stage, low_f, high_f):
        print("\n" + "------------------------------- Running Paikro Algorithm ----------------------------------")

        self.epoch_len = edf_file.epoch_len
        self.filter_stage = filter_stage
        self.low_f = low_f
        self.high_f = high_f

        self.file_id = edf_file.file_id
        self.signal_frequency = edf_file.signal_frequency
        self.starttime = edf_file.starttime
        self.collection_duration = edf_file.collection_duration

        self.ecg_raw = edf_file.ecg_raw
        self.raw_timestamps = edf_file.raw_timestamps

        self.ecg_deriv = None
        self.ecg_filtered = None
        self.ecg_squared = None
        self.peak_indices = None
        self.peak_values = None
        self.rr_intervals = []
        self.rr_hr = []

        self.corrected_peaks = []  # peaks that get removed in check process
        self.beat_timestamps = []
        self.epoch_hr = []
        self.epoch_timestamps = []
        self.epoch_start_stamp = []
        self.epoch_end_stamp = []
        self.epoch_beat_tally = []

        self.avg_hr = 0  # calculated from epoch_hr
        self.avg_rr_hr = 0  # calculated from beat-to-beat RR-intervals
        self.min_hr = 0
        self.max_hr = 0

        # Runs methods
        self.differentiate_ecg()
        self.square_data()
        self.detect_peaks()
        self.process_rr_hr()
        self.process_epochs()

    @staticmethod
    def bandpass_filter(dataset, lowcut, highcut, signal_freq, filter_order):
        """Method that creates bandpass filter to ECG data."""

        # Filter characteristics
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, dataset)

        return y

    def differentiate_ecg(self):
        """Uses numpy 'ediff1d' function to differentiate raw data."""

        if self.filter_stage == "raw":
            t0_filt = datetime.now()

            ecg_data = self.ecg_raw

            print("\n" + "Filtering raw data (8-18Hz, 2nd order bandpass filter).")
            self.ecg_filtered = self.bandpass_filter(ecg_data,
                                                     lowcut=self.low_f, highcut=self.high_f, filter_order=2,
                                                     signal_freq=self.signal_frequency)

            t1_filt = datetime.now()

            print("Data has been filtered. Took {} seconds.".format(round((t1_filt - t0_filt).seconds), 2))

        t0_diff = datetime.now()

        print("\n" + "Differentiating raw ECG signal...")

        self.ecg_deriv = np.ediff1d(self.ecg_raw)

        t1_diff = datetime.now()

        print("ECG signal differentiated. Took {} seconds.".format(round((t1_diff-t0_diff).seconds), 2))

    def square_data(self):
        """Squares filtered + differentiated data AND runs filtering method."""

        if self.filter_stage == "differentiated":
            t0_filt = datetime.now()

            ecg_data = self.ecg_deriv

            print("\n" + "Filtering differentiated data ({}-{}Hz, 2rd order bandpass filter).".
                  format(self.low_f, self.high_f))

            self.ecg_filtered = self.bandpass_filter(ecg_data,
                                                     lowcut=self.low_f, highcut=self.high_f, filter_order=2,
                                                     signal_freq=self.signal_frequency)

            t1_filt = datetime.now()

            print("Data has been filtered. Took {} seconds.".format(round((t1_filt-t0_filt).seconds), 2))

        t0_square = datetime.now()

        print("\n" + "Cubing data...")
        self.ecg_squared = self.ecg_filtered ** 3

        t1_square = datetime.now()
        print("Data has been squared. Took {} seconds.".format(round((t1_square-t0_square).seconds), 2))

    def detect_peaks(self):
        """Uses PeakUtils package to detect peaks in squared ECG signal. Handles spikes by reducing their value to
           prevent false negatives on other peaks due to thresholding system."""

        t0_peak = datetime.now()

        print("\n" + "Detecting QRS peaks...")

        # Filtering differentiated data first
        for i in range(len(self.ecg_deriv)):
            if self.ecg_deriv[i] >= 200:
                self.ecg_deriv[i] = 200
            else:
                self.ecg_deriv[i] = 0

        self.peak_indices = peakutils.indexes(self.ecg_deriv, thres=0.3, min_dist=5)
        self.peak_values = self.ecg_squared[self.peak_indices]

        t1_peak = datetime.now()
        print("QRS peaks detected. Found {} peaks. Took {} seconds.".format(len(self.peak_indices),
                                                                            round((t1_peak - t0_peak).seconds), 2))

    def process_rr_hr(self):
        """Calculates beat-to-beat HR using time between consecutive R-R intervals."""

        t0_rr = datetime.now()

        print("\n" + "Processing HR from consecutive RR-intervals...")

        # Calculates HR using converted time interval between QRS peak indices
        for beat1, beat2 in zip(self.peak_indices[:], self.peak_indices[1:]):
            # Does not use peaks less than 250ms apart
            if (beat2-beat1) > self.signal_frequency/4:
                self.rr_hr.append(round(60 / ((beat2 - beat1) / self.signal_frequency), 1))
                self.rr_intervals.append((beat2 - beat1) / self.signal_frequency)

        self.avg_rr_hr = round(stats.mean(self.rr_hr), 1)

        # Converts indices to datetime timestamps using collection_starttime and sample frequency
        # Also removes peaks less than 333 ms apart (HR >= 180 bpm)
        for peak in self.peak_indices:
            self.beat_timestamps.append(self.starttime + timedelta(seconds=peak / self.signal_frequency))

        t1_rr = datetime.now()

        print("RR-interval HR processed. Took {} seconds.".format(round((t1_rr - t0_rr).seconds), 2))

    def process_epochs(self):
        """Calculates average HR in each epoch using window length and number of beats in that window."""

        t0_epoch = datetime.now()

        print("\n" + "Epoching HR data...")

        # Creates list of timestamps corresponding to starttime + epoch_len for each epoch
        for i in range(0, int(self.collection_duration / self.epoch_len)):
            timestamp = self.beat_timestamps[0] + timedelta(seconds=i * self.epoch_len)
            self.epoch_timestamps.append(timestamp)

        # Loops through epoch start timestamps
        for epoch_start, epoch_end in zip(self.epoch_timestamps[:], self.epoch_timestamps[1:]):

            # Tally for number of beats in current epoch
            epoch_beat_tally = 0

            # Loops through heartrate_timestamps and counts beats that fall within current epoch
            for beat_index, beat_stamp in enumerate(self.beat_timestamps):

                if epoch_start <= beat_stamp < epoch_end:
                    epoch_beat_tally += 1

                # Gets timestamps of first and last beats in epoch
                if beat_stamp >= epoch_end:
                    self.epoch_start_stamp.append(self.beat_timestamps[beat_index - epoch_beat_tally])

                    self.epoch_end_stamp.append(self.beat_timestamps[beat_index])

                    break

            self.epoch_beat_tally.append(epoch_beat_tally)

        for start, stop, tally in zip(self.epoch_start_stamp, self.epoch_end_stamp, self.epoch_beat_tally):
            duration = (stop-start).seconds

            self.epoch_hr.append(round((tally - 1) * 60 / duration, 1))

        # Descriptive values from epoch_hr
        # Calculates average HR (non-weighted) from epoch_hr for values within "valid" range
        hr_sum_tally = 0
        hr_valid_epoch_tally = 0

        for hr in self.epoch_hr:
            if 40 < hr < 180:
                hr_sum_tally += hr
                hr_valid_epoch_tally += 1

        self.avg_hr = round(hr_sum_tally / hr_valid_epoch_tally, 1)

        t1_epoch = datetime.now()

        # Prints epoch HR data summary
        print("Epoched HR calculated. Took {} seconds.".format(round((t1_epoch - t0_epoch).seconds), 2))
        print("     Found {} epochs of length {} seconds.".format(len(self.epoch_hr), self.epoch_len))

        print("     {} heart beats detected.".format(len(self.peak_indices)))
        print("     Average heart rate: ~ {} bpm".format(self.avg_hr))

        self.max_hr = max(self.epoch_hr)
        self.min_hr = min(self.epoch_hr)

    def plot_data(self):
        """Plot with subplots that share x-axis. Plot 1: raw ECG with approximate peak locations marked.
           Plot 2: data upon which peak detection is run with peaks marked. Plot 3: Epoched HR"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col")

        # Raw data
        ax1.plot(self.raw_timestamps[::3], self.ecg_filtered[::3], color="red")
        ax1.plot([self.raw_timestamps[i] for i in self.peak_indices],
                 [self.ecg_filtered[i] for i in self.peak_indices],
                 color="black", linestyle="", marker="o", markeredgecolor="black", markerfacecolor="red", markersize=4)
        ax1.legend(loc="upper left", labels=["Filtered", "Peaks"])
        ax1.set_ylabel("Voltage (mV)")
        ax1.set_title("{}({} hours): filter run on {} data".
                      format(self.file_id, round(self.collection_duration / 60 / 60, 2), self.filter_stage))

        # Squared, filtered and differentiated data
        ax2.plot(self.raw_timestamps[:-1:3], self.ecg_deriv[::3], color="#459ACD")

        ax2.plot([self.raw_timestamps[i] for i in self.peak_indices],
                 self.peak_values, color="#459ACD", linestyle="",
                 marker="o", markeredgecolor="black", markerfacecolor="#459ACD", markersize=4)

        ax2.legend(loc="upper left", labels=["Squared", "Peaks"])
        ax2.set_ylabel("[dV/dt]^2")

        ax3.plot(self.beat_timestamps[0:len(self.rr_hr)], self.rr_hr, color="black", linewidth=0.5)

        ax3.plot(self.epoch_timestamps[0:len(self.epoch_hr)], self.epoch_hr, color="red",
                 marker="o", markeredgecolor="black", markerfacecolor="red", markersize=5)

        ax3.legend(loc="upper left", labels=["Beat-to-beat HR (avg = {} bpm".format(self.avg_rr_hr),
                                             "{}-sec epoch HR (avg = {} bpm)".format(self.epoch_len, self.avg_hr)])

        ax3.set_ylabel("HR (bpm)")
        ax3.set_ylim(0, 180)

        plt.xticks(rotation=45, size=6)
        ax3.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_locator(locator)

        plt.show()


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- RUNNING SCRIPT ---------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
input_file = "O:\\Data\\ReMiNDD\\Raw data\\Bittium\\OND06_SBH_1039_01_SE01_GABL_BF_02.EDF"
# input_file = "/Users/kyleweber/Desktop/Test Data/ECG Files/Kyle_BF.EDF"

#if __name__ == "__main__":

t0 = datetime.now()

edf_file = EdfFile(file=input_file, crop=[60, 120], epoch_len=15)
paikro_data = PaikroAlgorithm(filter_stage="differentiated", low_f=8, high_f=18)
# paikro_data_2 = PaikroAlgorithm(filter_stage="differentiated", low_f=8, high_f=18)

# comp = Compare(plot=True)

t1 = datetime.now()
print("\n" + "--------------------------------------------------------------------------------------------------| ")
print("Program complete. Run time = {} seconds.".format(round((t1-t0).seconds), 2))

paikro_data.plot_data()
