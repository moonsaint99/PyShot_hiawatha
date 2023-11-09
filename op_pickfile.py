import csv
import numpy as np
import copy

class opPickfile:
    def __init__(self, filename, depth=477):
        self.filename = filename
        # Import the csv, where each line is offset, time, amplitude, std
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.data = np.array(list(reader))
        self.dist = np.abs(self.data[:, 0].astype(np.float32))
        self.time = self.data[:, 1].astype(np.float32)
        self.amplitude_original = self.data[:, 2].astype(np.float32)
        self.amplitude = self.amplitude_original
        self.std = self.data[:, 3].astype(np.float32)
        self.angle = np.arcsin(self.dist / (2 * depth))

        # For compatibility with anything expecting pickfiles from su_pickfile.py
        self.max = self.amplitude
        self.tmax = self.time
        self.min = self.amplitude
        self.tmin = self.time
        self.abs = self.amplitude
        self.tabs = self.time
        self.max_no_outliers = self.amplitude
        self.tmax_no_outliers = self.time
        self.min_no_outliers = self.amplitude
        self.tmin_no_outliers = self.time
        self.abs_no_outliers = self.amplitude
        self.dist_no_outliers = self.dist

    def deviation(self, deviations):
        deviated_pickfile = copy.deepcopy(self)
        deviated_pickfile.amplitude = self.amplitude + self.std*deviations
        return deviated_pickfile