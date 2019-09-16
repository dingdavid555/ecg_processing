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
# ----------------------------------------------- HEARTPY ALGORITHM --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class HeartPyAlgorithm(object):

    def __init__(self, resample_factor, low_f, high_f):

        print("\n" + "------------------------------- Running HeartPy Algorithm -------------------------------")

        self.epoch_len = edf_file.epoch_len
        self.signal_frequency = edf_file.signal_frequency
        self.starttime = edf_file.starttime
        self.collection_duration = edf_file.collection_duration
        self.low_f = low_f
        self.high_f = high_f

        self.ecg_raw = edf_file.ecg_raw
        self.raw_timestamps = edf_file.raw_timestamps

        self.ecg_filtered = None

        self.wd = None  # dictionary of results
        self.summary = None  # summary metrics

        self.resample_factor = resample_factor
        self.ecg_resampled = None  # upsamples signal to help peak detection
        self.ecg_scaled = None
        self.corrected_peaks = []  # peaks that get removed in check process
        self.beat_timestamps = []
        self.epoch_hr = []
        self.epoch_timestamps = edf_file.epoch_timestamps
        self.epoch_start_stamp = []
        self.epoch_end_stamp = []
        self.epoch_beat_tally = []

        self.min_hr = 0
        self.max_hr = 0

        self.t0 = datetime.now()

        self.filter_ecg()
        self.process_ecg()
        self.process_epochs()

        self.t1 = datetime.now()

        print("\n" + "HeartPy processing complete. Total time = {} seconds.".
              format(round((self.t1-self.t0).seconds), 1))

    def filter_ecg(self):
        """Applies a 0.5-30Hz, 2nd order bandpass filter to raw data."""

        print("\n" + "Filtering data with {}-{}Hz, 2nd order bandpass filter...".format(self.low_f, self.high_f))

        # Runs bandpass filter on data
        self.ecg_filtered = heartpy.filter_signal(data=[float(i) for i in self.ecg_raw],
                                                  cutoff=[self.low_f, self.high_f],
                                                  sample_rate=self.signal_frequency,
                                                  filtertype="bandpass", order=2)
        print("Filtering complete.")

    def process_ecg(self):
        """Runs heartpy.process on up-sampled data and creates a timestamp for each detected beat"""

        print("\n" + "Detecting peaks using the HeartPy algorithm...")

        # Upsamples data by a factor of 4 to improve peak index resolution
        self.ecg_resampled = resample(self.ecg_filtered, len(self.ecg_filtered) * self.resample_factor)

        self.ecg_scaled = heartpy.scale_data([float(i) for i in self.ecg_resampled])

        # Processes data using standard heartpy processing
        self.wd, self.summary = heartpy.process(hrdata=self.ecg_scaled,
                                                sample_rate=self.signal_frequency*self.resample_factor)

        # Retrieves peaks that got removed in check process
        for peak in self.wd["peaklist"]:
            if peak not in self.wd["peaklist_cor"]:
                self.corrected_peaks.append(peak)

        # Creates a timestamp for every beat using peak indexes
        for peak in self.wd["peaklist_cor"]:
            self.beat_timestamps.append(self.starttime +
                                        timedelta(seconds=peak / (self.signal_frequency * self.resample_factor)))

        print("Peak detection complete.")

    def process_epochs(self):
        """Calculates average HR in each epoch using epoch length and number of beats in that epoch."""

        t0_epoch = datetime.now()

        print("\n" + "Epoching HR data...")

        # Loops through epoch start timestamps
        for epoch_start, epoch_end in zip(self.epoch_timestamps[:], self.epoch_timestamps[1:]):

            # Tally for number of beats in current epoch
            epoch_beat_tally = 0

            # Loops through rr_hr_timestamps and counts beats that fall within current epoch
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
            duration = (stop - start).seconds

            self.epoch_hr.append(round((tally - 1) * 60 / duration, 1))

        t1_epoch = datetime.now()

        # Prints epoch HR data summary
        print("Epoched HR calculated. Took {} seconds.".format(round((t1_epoch-t0_epoch).seconds), 2))
        print("     Found {} epochs of length {} seconds.".format(len(self.epoch_hr), self.epoch_len))

        print("     {} heart beats detected.".format(len(self.wd["peaklist_cor"])))
        print("     Average heart rate: ~ {} bpm".format(round(self.summary["bpm"], 1)))

        self.max_hr = max(self.epoch_hr)
        self.min_hr = min(self.epoch_hr)

    def plot_peaks(self):
        """Runs heartpy.plotter for visualizing filtered data and peaks."""

        heartpy.plotter(working_data=self.wd, measures=self.summary)

    def plot_epochs(self):
        """Plots epoched HR and filtered data with detected peaks."""

        fig, (ax1, ax2) = plt.subplots(2, sharex="col")

        # Epoched data
        ax1.plot(self.epoch_timestamps[0:len(self.epoch_hr)], self.epoch_hr,
                 color="black", marker="o", markerfacecolor="red", markeredgecolor="black", markersize=4)
        ax1.set_ylim(-5, 180)
        ax1.set_ylabel("HR (bpm)")
        ax1.set_title("HeartPy Algorithm: {}-Sec Epoch HR Data".format(self.epoch_len))

        # Peak data
        ax2.plot(self.raw_timestamps[::3], self.ecg_filtered[::3], color="black")

        ax2.plot([self.raw_timestamps[int(i/self.resample_factor)] for i in self.wd["peaklist_cor"][0:]],
                 [self.ecg_filtered[int(i/self.resample_factor)] for i in self.wd["peaklist_cor"][0:]],
                 linestyle="", marker="o", markerfacecolor="green", markeredgecolor="black", markersize=4)

        ax2.plot([self.raw_timestamps[int(i/self.resample_factor)] for i in self.corrected_peaks],
                 [self.ecg_filtered[int(i/self.resample_factor)] for i in self.corrected_peaks],
                 linestyle="", marker="x", markeredgecolor="red", markersize=4)

        ax2.legend(loc="upper left", labels=["Filtered data", "Peaks (n={})".format(len(self.wd["peaklist_cor"])),
                                             "False peaks (n={})".format(len(self.corrected_peaks))])

        ax2.set_ylabel("Voltage")

        # Formatting
        plt.xticks(rotation=45, size=6)
        ax2.xaxis.set_major_formatter(xfmt)
        ax2.xaxis.set_major_locator(locator)

        plt.show()


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

        print("\n" + "Squaring data...")
        self.ecg_squared = self.ecg_filtered ** 2

        t1_square = datetime.now()
        print("Data has been squared. Took {} seconds.".format(round((t1_square-t0_square).seconds), 2))

    def detect_peaks(self):
        """Uses PeakUtils package to detect peaks in squared ECG signal. Handles spikes by reducing their value to
           prevent false negatives on other peaks due to thresholding system."""

        t0_peak = datetime.now()

        print("\n" + "Detecting QRS peaks...")

        # Peak detection using PeakUtils package in 15-second windows
        for start_index in range(0, len(self.ecg_raw), self.signal_frequency * self.epoch_len):

            # Data section to use
            current_data = self.ecg_squared[start_index:start_index + self.signal_frequency * self.epoch_len - 1]

            # Scales spikes down to prevent outrageous values that ruin peak detection
            voltage_sd = stats.stdev(current_data)
            voltage_mean = stats.mean(current_data)

            for i, datapoint in enumerate(current_data):
                if datapoint >= voltage_mean + voltage_sd:
                    current_data[i] = int(voltage_mean + voltage_sd)

            # Detects peak in 1-minute window
            detected_peaks_indices = peakutils.indexes(current_data, thres=0.3,
                                                       min_dist=int(self.signal_frequency / 3),
                                                       thres_abs=False)

            # Corrects indices to account for previous 1-minute windows
            detected_peaks_indices = detected_peaks_indices + start_index

            # Converts self.peak_indices to numpy array on first loop
            if start_index == 0:
                self.peak_indices = detected_peaks_indices

            # Concatenates new window peaks with existing peaks
            if start_index != 0:
                self.peak_indices = np.concatenate((self.peak_indices, detected_peaks_indices))

        # Gets squared ECG values of each peak
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
        ax2.plot(self.raw_timestamps[:-1:3], self.ecg_squared[::3], color="#459ACD")

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
# ---------------------------------------------- PAN-TOMPKINS ALGORITHM ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class PanTompkinsAlgorithm(object):

    # change outlier removal to filtered data

    def __init__(self):
        self.epoch_len = edf_file.epoch_len

        self.file_id = edf_file.file_id
        self.signal_frequency = edf_file.signal_frequency
        self.starttime = edf_file.starttime
        self.collection_duration = edf_file.collection_duration

        self.ecg_raw = edf_file.ecg_raw
        self.raw_timestamps = edf_file.raw_timestamps
        self.epoch_timestamps = edf_file.epoch_timestamps

        # Number of samples windows are integrated over, integer
        self.integration_window = int(self.signal_frequency * 3 / 20)  # n samples over which data is integrated

        # Peak detection parameters
        self.peak_threshold = 0.15  # peak threshold, normalized
        self.peak_spacing = int(self.signal_frequency / 4)  # n samples for minimum consecutive peaks
        self.refractory_period = int(self.signal_frequency / 4)  # n samples for refractory period

        # See Pan-Tompkins article
        self.qrs_peak_filtering_factor = 0.125  # leave this alone
        self.noise_peak_filtering_factor = 0.875  # leave this alone
        self.qrs_noise_diff_weight = 0.25  # leave this alone

        self.ecg_filtered = None  # filtered ECG data
        self.differentiated_ecg_measurements = None  # filtered and differentiated ECG data
        self.ecg_squared = None  # filtered, differentiated and squared ECG data
        self.ecg_integrated = None  # filtered, differentiated, squared and integrated ECG data
        self.detected_peaks_indices = None  # indices in integrated data of each potential peak
        self.detected_peaks_values = None  # values from integrated data at each potential peak
        self.peak_candidate = None

        # Peak detection variables. Do not touch
        self.qrs_peak_value = 0.0
        self.noise_peak_value = 0.0
        self.threshold_value = 0.0

        # Data objects
        self.rr_hr = []  # beat-to-beat heart rates
        self.rr_hr_timestamps = []  # beat-to-beat timestamps corresponding to first of the 2 beats
        self.epoch_hr = []  # average HR in each epoch

        # Descriptive values
        self.avg_hr = 0.0  # average of epoched HRs (not weighted)
        self.max_hr = 0.0  # max epoched HR value
        self.min_hr = 0.0  # min epoched HR value

        # Numpy arrays that store calculated peak indices
        self.qrs_peaks_indices = np.array([], dtype=int)  # final data for detected peaks
        self.noise_peaks_indices = np.array([], dtype=int)  # noise peaks that are removed

        # Runs functions
        self.detect_peaks(threshold=self.peak_threshold)
        self.detect_qrs()
        self.process_beattobeat()

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """Creates bandpass filter for ECG data."""

        # Filter characteristics
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

    def findpeaks(self, data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """

        # Data object set-up
        data_len = data.size  # Number of data points
        x = np.zeros(data_len + 2 * spacing)  # Creates placeholder list of 0's
        x[:spacing] = data[0] - 1.e-6  # Changes first spacing number of data points to 1e-6
        x[-spacing:] = data[-1] - 1.e-6  # Changes last spacing number of data points to 1e-6
        x[spacing:spacing + data_len] = data  # Adds integrated data to middle of 0's list
        self.peak_candidate = np.zeros(data_len)
        self.peak_candidate[:] = True  # Changes zeros to True (binary 1)

        for s in range(int(spacing)):
            start = spacing - s - 1
            h_b = x[start: start + data_len]  # before

            start = spacing
            h_c = x[start: start + data_len]  # central

            start = spacing + s + 1
            h_a = x[start: start + data_len]  # after

            self.peak_candidate = np.logical_and(self.peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(self.peak_candidate)  # Returns indices where peak_candidate is True in long format
        ind = ind.reshape(ind.size)  # Changes array to wide format

        if limit is not None:
            ind = ind[data[ind] > limit]  # Gets indices of values in integrated data > threshold
        return ind

    def detect_peaks(self, threshold):
        """
        -Runs Kyle's outlier detection method of cropping data points outside ± 3 SD
        -Performs Pan-Tompkins differentiation and squaring transformations
        -Runs findpeaks method
        -Creates dataset for detected peaks
        """

        # Raw data for use in this method
        ecg_measurements = self.ecg_raw

        # Raw data filter: {lowcut-highcut} Hz band pass filter.
        print("\n" + "Filtering data with a 3-15Hz, order 1 bandpass filter.")

        # Object for storing filtered data
        self.ecg_filtered = self.bandpass_filter(ecg_measurements, lowcut=3,
                                                 highcut=15,
                                                 signal_freq=self.signal_frequency,
                                                 filter_order=2)

        print("Filtering complete.")

        # Kyle's outlier removal on filtered data
        print("\n" + "Removing outlier data points from filtered data...")

        # Calculates SD of filtered ECG data
        voltage_sd = int(stats.stdev(self.ecg_filtered))

        # Loops through filtered ECG data
        for i, datapoint in enumerate(self.ecg_filtered):

            # If voltage is outside + 3SD range, crops its value to + 3SD
            if datapoint >= 3 * voltage_sd:
                self.ecg_filtered[i] = int(3 * voltage_sd)

            # If voltage is outside - 3SD range, crops its value to - 3SD
            if datapoint <= -3 * voltage_sd:
                self.ecg_filtered[i] = int(-3 * voltage_sd)

        print("Outliers removed from filtered data.")

        self.ecg_filtered[:5] = self.ecg_filtered[5]

        # Takes derivative (provides QRS slope information)
        self.differentiated_ecg_measurements = np.ediff1d(self.ecg_filtered)

        # Squares derivatives (intensifies values received in derivative)
        # PAN-TOMPKINS EQUATION # 10
        self.ecg_squared = self.differentiated_ecg_measurements ** 2

        # Moving-window integration
        # PAN-TOMPKINS EQUATION #11
        self.ecg_integrated = np.convolve(self.ecg_squared, np.ones(self.integration_window))

        # Creates dataset of peaks and peak heights
        # Calls peak detection method which is run on integrated ECG data

        print("\n" + "Detecting peaks...")

        self.detected_peaks_indices = self.findpeaks(data=self.ecg_integrated,
                                                     limit=self.peak_threshold,
                                                     spacing=self.peak_spacing)

        # Peak detection using PeakUtils package
        """
        self.detected_peaks_indices = peakutils.indexes(self.ecg_integrated,
                                                        thres=threshold, min_dist=int(self.peak_spacing/2),
                                                        thres_abs=False)  # Set value if True, normalized if False"""

        # Creates list of integrated ECG values that correspond to detected peaks
        self.detected_peaks_values = self.ecg_integrated[self.detected_peaks_indices]

        print("Peaks detected. Found {} potential QRS peaks.".format(len(self.detected_peaks_indices)))

    def detect_qrs(self):
        """Detects QRS peaks."""

        print("\n" + "Removing noise peaks from potential QRS peaks...")
        # Classifies detected peak as either valid QRS or as noise
        for detected_peak_index, detected_peaks_value in zip(self.detected_peaks_indices, self.detected_peaks_values):

            # Prevents error for first index
            try:
                last_qrs_index = self.qrs_peaks_indices[-1]

            except IndexError:
                last_qrs_index = 0

            # Loops through consecutive peak indexes to check for > refractory period spacing
            # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected
            if detected_peak_index - last_qrs_index > self.refractory_period or not self.qrs_peaks_indices.size:

                # Peak must be classified either as a noise peak or a QRS peak
                # To be classified as a QRS peak, it must exceed dynamically set threshold value
                if detected_peaks_value > self.threshold_value:
                    self.qrs_peaks_indices = np.append(self.qrs_peaks_indices, detected_peak_index)

                    # Adjust QRS peak value used later for setting QRS-noise threshold
                    # PAN-TOMPKINS EQUATION #12
                    self.qrs_peak_value = self.qrs_peak_filtering_factor * detected_peaks_value + \
                        (1 - self.qrs_peak_filtering_factor) * self.qrs_peak_value

                # If detected peak <= 200ms after previous peak or below threshold, it's considered noise
                else:
                    self.noise_peaks_indices = np.append(self.noise_peaks_indices, detected_peak_index)

                    # Adjust noise peak value used later for setting QRS-noise threshold.
                    # PAN-TOMPKINS EQUATION #13
                    self.noise_peak_value = self.noise_peak_filtering_factor * detected_peaks_value + \
                        (1 - self.noise_peak_filtering_factor) * self.noise_peak_value

                # Adjusts QRS-noise threshold value based on previously detected QRS or noise peaks value
                # PAN-TOMPKINS EQUATION #14
                self.threshold_value = self.noise_peak_value + \
                    self.qrs_noise_diff_weight * (self.qrs_peak_value - self.noise_peak_value)

        print("Noise peaks removed. Removed {} noise peaks.".format(len(self.detected_peaks_indices) -
                                                                    len(self.qrs_peaks_indices)))

        # Converts indices to datetime timestamps using collection_starttime and sample frequency
        for i in self.qrs_peaks_indices:
            self.rr_hr_timestamps.append(self.starttime + timedelta(seconds=i / self.signal_frequency))

    def process_beattobeat(self):
        """Calculates beat-to-beat HR using time between consecutive R-R intervals."""

        # Calculates HR using converted time interval between QRS peak indices
        for beat1, beat2 in zip(self.qrs_peaks_indices[:], self.qrs_peaks_indices[1:]):
            self.rr_hr.append(round(60 / ((beat2 - beat1) / self.signal_frequency), 1))


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------- COMPARISON CLASS ---------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class Compare(object):

    def __init__(self, plot):
        self.plot = plot

        self.epoch_len = hp_data.epoch_len
        self.hp = hp_data.epoch_hr
        self.hp_avg = round(stats.mean(self.hp), 1)
        self.paikro = paikro_data.epoch_hr
        self.paikro_avg = round(stats.mean(self.paikro), 1)

        self.epoch_timestamps = hp_data.epoch_timestamps

        if self.plot:
            self.plot_comparison()

    def plot_comparison(self):

        comp_plot = plt.gca()

        plt.plot(self.epoch_timestamps[0:len(self.paikro)], self.paikro,
                 color="black", marker="o", markerfacecolor="black", markeredgecolor="black")

        plt.plot(self.epoch_timestamps[0:len(self.hp)], self.hp,
                 color="red", marker="o", markerfacecolor="red", markeredgecolor="black")

        plt.xlabel("Timestamps")
        plt.ylabel("Epoch HR (bpm)")
        plt.title("Epoched HR Comparison ({}-sec epochs)".format(self.epoch_len))

        plt.ylim(0, 180)

        plt.legend(loc="upper left", labels=["Paikro (avg = {} bpm)".format(self.paikro_avg),
                                             "HeartPy (avg = {} bpm)".format(self.hp_avg)])

        plt.xticks(rotation=45, size=6)
        comp_plot.xaxis.set_major_formatter(xfmt)
        comp_plot.xaxis.set_major_locator(locator)

        plt.plot()


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- RUNNING SCRIPT ---------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

input_file = "/Users/kyleweber/Desktop/Test Data/ECG Files/OF_Bittium_HR.EDF"
# input_file = "/Users/kyleweber/Desktop/Test Data/ECG Files/Kyle_BF.EDF"

if __name__ == "__main__":

    t0 = datetime.now()

    edf_file = EdfFile(file=input_file, crop=[60, 120], epoch_len=15)
    hp_data = HeartPyAlgorithm(resample_factor=8, low_f=8, high_f=18)
    # paikro_data = PaikroAlgorithm(filter_stage="differentiated", low_f=8, high_f=18)
    # pt_data = PanTompkinsAlgorithm()

    # comp = Compare(plot=True)

    t1 = datetime.now()
    print("\n" + "--------------------------------------------------------------------------------------------------| ")
    print("Program complete. Run time = {} seconds.".format(round((t1-t0).seconds), 2))
