import obspy as op
from obspy.io.segy.core import _read_segy
import numpy as np
import os
from load_segy import loadshots


# Specify the directory where your .su files are stored.
directory_path = "./lithic_shots/"

# Load all .su files in the directory into a dictionary of ObsPy Stream objects.
seismic_streams = loadshots(directory_path)

# To access a specific Stream by filename, you would do:
# specific_stream = seismic_streams["72_wind.su"]

# And to access a specific Trace within that Stream, you would do:
# specific_trace = specific_stream[0]  # This gets the first Trace in the Stream

