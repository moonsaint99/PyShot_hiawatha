import csv
import numpy as np

class opPickfile:
    def __init__(self, filename):
        self.filename = filename
        # Import the csv, where each line is offset, time, amplitude
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.data = np.array(list(reader))
        self.dist = self.data[:, 0].astype(np.float)
        self.time = self.data[:, 1].astype(np.float)
        self.amplitude = self.data[:, 2].astype(np.float)

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
