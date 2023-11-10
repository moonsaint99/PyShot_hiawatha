import copy

import obspy as op
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import csv
import time
from load_segy import loadshots

streamdict = loadshots("lithic_smallstack_streams/")
for filename, stream in streamdict.items():
    starttime = stream[0].stats.starttime
    stream.trim(starttime, starttime + 0.3)


# This script performs FK filtering of a stream object, where each trace is a different offset

# This function computes the FK transform of a stream object
def FK(stream, filename):
    # stream is an ObsPy stream object
    data = np.array([trace.data for trace in stream])
    # fmin is the minimum frequency to keep
    # fmax is the maximum frequency to keep

    # Get the sampling rate
    dt = stream[0].stats.delta

    # Get the spatial interval
    dx = 5

    # Get the number of samples in each trace
    nsamples_time = len(stream[0])

    # Get the number of traces in the stream
    nsamples_space = len(stream)

    # Frequency axis
    faxis = np.fft.fftfreq(nsamples_time, d=dt)
    kaxis = np.fft.fftfreq(nsamples_space, d=dx)

    # Shift zero to center
    faxis = np.fft.fftshift(faxis)
    kaxis = np.fft.fftshift(kaxis)

    # Compute the 2D FFT
    data_fk = np.fft.fftshift(np.fft.fft2(data))

    # Return the FK spectrum and the frequency and wavenumber axes
    return data_fk, faxis, kaxis


# This function plots the FK spectrum of a stream object
def FK_plot(FK_output, fmin, fmax):
    data_fk = FK_output[0]
    faxis = FK_output[1]
    kaxis = FK_output[2]


    # Plot the FK spectrum
    plt.figure()
    plt.imshow(np.abs(data_fk), extent=[kaxis[0], kaxis[-1], fmin, fmax], aspect='auto', cmap='RdYlGn')
    plt.xlabel('k (1/m)')
    plt.ylabel('f (Hz)')
    plt.title('FK Spectrum for ' + filename)
    plt.colorbar()
    plt.show()


# This function produces a velocity mask for an FK spectrum
def FK_mask(FK_output, vmin, vmax):
    data_fk = FK_output[0]
    faxis = FK_output[1]
    kaxis = FK_output[2]

    # Create a mask for the FK spectrum
    freq, kx = np.meshgrid(faxis, kaxis)
    velocity = freq/kx
    mask = np.logical_and(velocity > vmin, velocity < vmax)

    # Apply the mask to the FK spectrum
    data_fk_masked = np.where(mask, 1, data_fk)

    return data_fk_masked, faxis, kaxis


# This function produces the IFFT of an FK spectrum, filters it, and puts the result into
# a new obspy stream object
def FK_IFFT(stream, filename, vmin, vmax):
    FK_output = FK(stream, filename)
    FK_masked = FK_mask(FK_output, vmin, vmax)
    data_fk_masked = FK_masked[0]
    data_ifk = np.fft.ifft2(np.fft.ifftshift(data_fk_masked))
    data_ifk = np.real(data_ifk)
    data_ifk = np.transpose(data_ifk)
    data_ifk = np.require(data_ifk, dtype=np.float32)
    newstream = copy.deepcopy(stream)
    for i in range(len(newstream)):
        newstream[i].data = data_ifk[i]
    return newstream


for filename, stream in streamdict.items():
    FK_output = FK(stream, filename)
    FK_plot(FK_output, 0, 500)
    time.sleep(1)
    FK_masked = FK_mask(FK_output, 0, 2000)
    FK_plot(FK_masked, 0, 500)