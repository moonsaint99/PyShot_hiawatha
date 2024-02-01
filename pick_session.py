import obsPicker as opick
import obspy as op
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import os
import csv
from load_segy import loadshots
from stepgain import apply_step_gain
mpl.use('macosx')


aquifer_streams = loadshots("./aquifer_shotstack/")

for filename, stream in aquifer_streams.items():
    starttime = stream[0].stats.starttime
    stream.trim(starttime, starttime + 0.3)
    # stream = apply_step_gain(stream, 0.15, 1, 5)
    for trace in stream:
        trace.detrend('linear')
        trace.detrend('demean')
        # trace.filter('bandpass', freqmin=10, freqmax=500, corners=4, zerophase=True)
        trace.data = np.require(trace.data, dtype=np.float32)
    opick.Pick(stream, filename, os.path.splitext('aquifer_shotstack_picks/' + str(filename))[0]+'.csv')