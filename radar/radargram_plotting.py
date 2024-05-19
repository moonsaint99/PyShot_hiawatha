import impdar as imp
from impdar.lib import load
from impdar.lib import plot
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('QtAgg')
import h5py

line_1 = './CSARP/Data_20160512_02_009.mat'
line_1 = load.load_mcords.load_mcords_mat(line_1)

# We will calculate elevations throughout the radargram
# To do this, we start with the depths of every point on the radargram
# For each spatial point along the line, we can subtract depth from the picked surface elevation
# This leaves us with depth converted to geographic elevation throughout the radargram

# Surface elevation picks are in the CSVs for each line, along with coordinates

plot.plot_radargram(line_1, ydat='dual', cmap=plt.cm.bone)
plt.show()