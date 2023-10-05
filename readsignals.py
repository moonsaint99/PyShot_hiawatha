import obspy as op
from obspy.io.segy.core import _read_segy
import numpy as np
import os
from load_segy import loadshots
import matplotlib.pyplot as plt
import wf_gridsearch as wfgs
import wf_model as wfm
import multiprocessing
import time

def main():
    # Load all .su files in the directory into a dictionary of ObsPy Stream objects.
    lithic_streams = loadshots("./lithic_shots/")
    aquifer_streams = loadshots("./aquifer_shots/")

    # To access a specific Stream by filename, you would do:
    # specific_stream = seismic_streams["72_wind.su"]

    # And to access a specific Trace within that Stream, you would do:
    # specific_trace = specific_stream[0]  # This gets the first Trace in the Stream

    for filename, stream in lithic_streams.items():
        starttime = stream[0].stats.starttime
        stream.trim(starttime, starttime + 0.3)
    lithic_streams["33.su"].plot(type='section', time_down=True, fillcolors=('blue', 'red'), color='none', size=(800, 1600))

    for filename, stream in aquifer_streams.items():
        starttime = stream[0].stats.starttime
        stream.trim(starttime, starttime + 0.3)
    aquifer_streams["72_wind_nh.su"].plot(type='section', time_down=True, fillcolors=('blue', 'red'), color='none', size=(1000, 1600))

    # Stack all traces
    lithic_alltraces = op.Stream()
    for filename, stream in lithic_streams.items():
        lithic_alltraces += stream
    lithic_alltraces.stack(group_by='{distance}')
    stackfig = plt.figure(figsize=(15, 15))
    lithic_alltraces.plot(type='section', fig=stackfig, time_down=True, fillcolors=('blue', 'red'), color='none', size=(1200, 1600), offset_min=0, offset_max=115)

    ax = plt.gca()

    def refl_time(offset, angle, velocity, depth=405):
        refl_timing = 2*np.sqrt(
                                (depth ** 2) +
                                ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
        ) / velocity
        return refl_timing

    distances = np.linspace(0, 0.235, 1000)
    theoretical_times = refl_time(distances, 0, 3800, depth=477)
    ax.plot(distances, theoretical_times, color='black', linewidth=1)
    stackfig.show()

    dt = lithic_streams["33.su"][0].stats.delta
    t_array = np.arange(0, 0.3, dt)
    synthetic_source = wfm.ricker_wavelet(125, 0.025, dt)
    synthetic = wfm.compute_seismogram_2layer(t_array, synthetic_source[1], 477, [1000, 3630, 3630/2], [2700, 5800, 5800/1.82], 50, lithic_streams["33.su"][0].stats.delta)
    plt.plot(t_array, synthetic, color='black', linewidth=1)
    plt.show()


    # runtime_Start = time.time()
    # wfgs.depthvel_gridsearch_wf_plot(aquifer_streams, prior=[3630, 405])
    # print('Time taken: ' + str(time.time() - runtime_Start) + ' seconds')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()