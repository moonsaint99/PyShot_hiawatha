import numpy as np
import scipy as sp

class Pickfile:
    def __init__(self, filename, order, shotloc, outliers=0, maxrows=12, pvel=3800, pintercept=0, timecorrection=0, depth=477, spacing=5):
        self.filename = filename
        self.data = np.loadtxt(filename, skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12), max_rows=maxrows)
        self.trace = self.data[:,0]
        self.max = self.data[:,6]
        self.tmax = self.data[:,7]
        self.min = self.data[:,8]
        self.tmin = self.data[:,9]
        self.tmin_avg = np.average(self.tmin)
        self.abs = self.data[:,10]
        self.tabs= self.data[:,11]
        self.energy = self.data[:,12]
        self.polarity = self.max == self.abs
        # When curve-fitting an exponential to self.min, we need to remove
        # the largest three values because they are outliers
        self.min_outliers = np.argpartition(-self.min, -1*outliers)[-1*outliers:]
        self.min_outliers = -self.min_outliers
        # self.min_outliers = 0
        # The above attribute could be turned into a method
        # that accepts a number of outliers to discard.
        # If we did this, the code could intelligently discard
        # the fewest number of outliers necessary to get a good fit.
        self.min_no_outliers = np.delete(self.min, self.min_outliers)
        self.max_no_outliers = np.delete(self.max, self.min_outliers)
        self.tmax_no_outliers = np.delete(self.tmax, self.min_outliers)
        self.abs_no_outliers = np.delete(self.abs, self.min_outliers)
        self.tabs_no_outliers = np.delete(self.tabs, self.min_outliers)
        self.trace_no_outliers = np.delete(self.trace, self.min_outliers)
        if shotloc == 'unknown':
            # Compute the distance using a fixed velocity and average traveltime of the first arrival
            tavg = np.average(self.tmin) + pintercept/pvel
            distavg = tavg * pvel
            shotloc = np.average(5*self.trace) + distavg
        elif shotloc == 'secondary':
            shotloc = np.nan
        if order == 'decr':
            self.dist = abs(shotloc - (self.trace * spacing))
            self.dist_no_outliers = np.delete(self.dist, self.min_outliers)
        elif order == 'incr':
            self.dist = abs(shotloc + (self.trace * spacing))
            self.dist_no_outliers = np.delete(self.dist, self.min_outliers)
        self.angle = np.arctan(self.dist/depth)
        self.angle_no_outliers = np.arctan(self.dist_no_outliers/depth)
        self.tmin_no_outliers = np.delete(self.tmin, self.min_outliers)

    def fliptrace(self, tracenumber):
        newmax = self.min[tracenumber]
        newtmax = self.tmin[tracenumber]
        newmin = self.max[tracenumber]
        newtmin = self.tmax[tracenumber]
        newmax_no_outliers = self.min_no_outliers[tracenumber]
        newtmax_no_outliers = self.tmin_no_outliers[tracenumber]
        newmin_no_outliers = self.max_no_outliers[tracenumber]
        newtmin_no_outliers = self.tmax_no_outliers[tracenumber]
        self.max[tracenumber] = newmax
        self.tmax[tracenumber] = newtmax
        self.min[tracenumber] = newmin
        self.tmin[tracenumber] = newtmin
        self.max_no_outliers[tracenumber] = newmax_no_outliers
        self.tmax_no_outliers[tracenumber] = newtmax_no_outliers
        self.min_no_outliers[tracenumber] = newmin_no_outliers
        self.tmin_no_outliers[tracenumber] = newtmin_no_outliers