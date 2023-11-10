import numpy as np
import os
from load_segy import loadshots
import matplotlib as mpl
mpl.use('macosx')
import multiprocessing
import obsPicker as opick


def main():
    # Load all .su files in the directory into a dictionary of ObsPy Stream objects.
    lithic_streams = loadshots("./lithic_shots/")
    aquifer_streams = loadshots("./aquifer_shots/")
    aquifer_streamstack = loadshots("aquifer_shotstack/")
    lithic_streamstack = loadshots("lithic_shotstack/")
    lithic_smallstack_streams = loadshots("lithic_smallstack_streams/")

    # To access a specific Stream by filename, you would do:
    # specific_stream = seismic_streams["72_wind.su"]

    # And to access a specific Trace within that Stream, you would do:
    # specific_trace = specific_stream[0]  # This gets the first Trace in the Stream

    # for filename, stream in lithic_streams.items():
    #     starttime = stream[0].stats.starttime
    #     stream.trim(starttime, starttime + 0.3)
    # # lithic_streams["33.su"].plot(type='section', time_down=True, fillcolors=('blue', 'red'), color='none', size=(800, 1600))
    #
    for filename, stream in aquifer_streams.items():
        starttime = stream[0].stats.starttime
        stream.trim(starttime, starttime + 0.3)
        for trace in stream:
            trace.detrend('linear')
            trace.detrend('demean')
            # trace.filter('bandpass', freqmin=10, freqmax=200, corners=4, zerophase=True)
            trace.data = np.require(trace.data, dtype=np.float32)
        # stream.filter('bandpass', freqmin=50, freqmax=200, corners=4, zerophase=True)
        stream.write("segy_write/" + filename, format="SU")
        opick.Pick(stream, filename, os.path.splitext(str(filename))[0]+'.csv')
    # # aquifer_streams["72_wind_nh.su"].plot(type='section', time_down=True, fillcolors=('blue', 'red'), color='none', size=(1000, 1600))
    #
    # for filename, stream in lithic_streamstack.items():
    #     starttime = stream[0].stats.starttime
    #     stream.trim(starttime, starttime + 0.3)

    # for filename, stream in lithic_smallstack_streams.items():
    #     starttime = stream[0].stats.starttime
    #     stream.trim(starttime, starttime + 0.3)
    #     for trace in stream:
    #         trace.detrend('linear')
    #         trace.detrend('demean')
    #         # trace.filter('bandpass', freqmin=10, freqmax=300, corners=4, zerophase=True)
    #         trace.data = np.require(trace.data, dtype=np.float32)
    #     # stream.filter('bandpass', freqmin=50, freqmax=200, corners=4, zerophase=True)
    #     stream.write("segy_write/" + filename, format="SU")
    #     opick.Pick(stream, filename, os.path.splitext(str(filename))[0]+'.csv')



    # # Stack all traces
    # lithic_alltraces = op.Stream()
    # for filename, stream in lithic_streams.items():
    #     lithic_alltraces += stream
    # lithic_alltraces.stack(group_by='{distance}')
    # stackfig = plt.figure(figsize=(8, 12))
    # # lithic_alltraces.plot(type='section', fig=stackfig, time_down=True, fillcolors=('blue', 'red'), color='none', size=(1200, 1600), offset_min=0, offset_max=115)
    #
    # # Apply a step gain to lithic_alltraces
    # lithic_alltraces_sg = sg.apply_step_gain(lithic_alltraces, 0.15, 1, 20)
    #
    # lithic_alltraces_sg.plot(type='section', fig=stackfig, time_down=True, fillcolors=('blue', 'red'), color='none', size=(1200, 1600), offset_min=0, offset_max=115)
    #
    #
    # ax = plt.gca()

    # def refl_time(offset, angle, velocity, depth=477):
    #     refl_timing = 2*np.sqrt(
    #                             (depth ** 2) +
    #                             ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    #     ) / velocity
    #     return refl_timing

    # # distances = np.linspace(0, 0.235, 1000)
    # # theoretical_times = refl_time(distances, 0, 3630, depth=477)
    # # ax.plot(distances, theoretical_times, color='black', linewidth=1)
    # stackfig.show()
    #
    # # dt = lithic_streams["33.su"][0].stats.delta
    # # t_array = np.arange(0, 0.3, dt)
    # # synthetic = wfm.retrieve_seismogram_2layer(t_array, 477, [1000, 3630, 3630 / 2],
    # #                                            [2700, 5800, 5800 / 1.82], 50,
    # #                                            dt=lithic_streams["33.su"][0].stats.delta)
    # # plt.plot(t_array, synthetic, color='black', linewidth=1)
    # # plt.show()
    #
    # # runtime_Start = time.time()
    # # threelayer_depths = [406, 10]
    # # threelayer_depths_std = [0.5, 0.1]
    # # threelayer_rho = [917, 2120, 3300]
    # # threelayer_rho_std = [0, 20, 50]
    # # threelayer_vp = [3630, 1700, 6200]
    # # threelayer_vp_std = [10, 30, 50]
    # # threelayer_vs = [1815, 160, 6200/1.82]
    # # threelayer_vs_std = [25, 15, 25]
    # # init_params = [threelayer_depths, threelayer_rho, threelayer_vp, threelayer_vs]
    # # init_std = [threelayer_depths_std, threelayer_rho_std, threelayer_vp_std, threelayer_vs_std]
    # twolayer_depths = [409]
    # twolayer_depths_std = [0.05]
    # twolayer_rho = [917, 3300]
    # twolayer_rho_std = [0, 0.5]
    # twolayer_vp = [3630, 6200]
    # twolayer_vp_std = [0.1, 0.3]
    # twolayer_vs = [1815, 6200/1.82]
    # twolayer_vs_std = [0.1, 0.1]
    # init_params = [twolayer_depths, twolayer_rho, twolayer_vp, twolayer_vs]
    # init_std = [twolayer_depths_std, twolayer_rho_std, twolayer_vp_std, twolayer_vs_std]
    # # accepted_params = mh.metropolis_hastings(init_params, int(1e6), aquifer_streamstack, init_std, step_size=1e-13)
    # # with open('accepted_params.pkl', 'wb') as f:
    # #     pickle.dump(accepted_params, f)
    # # save accepted params to file
    # # current_time = str(time.time())
    # # np.save('accepted_params.npy', np.array(accepted_params))
    # twolayer = wfm.Model(twolayer_depths, twolayer_rho, twolayer_vp, twolayer_vs)
    # twolayer_synthetic = twolayer.get_synthetic(np.arange(0, 1, 0.00025), 50, dt=aquifer_streamstack["stack_output.su"][0].stats.delta)
    # # print('Time taken: ' + str(time.time() - runtime_Start) + ' seconds')
    # # plt.plot(np.arange(0, 1, 0.00025), twolayer_synthetic, color='black', linewidth=1)
    # plt.show()

    # runtime_Start = time.time()
    # wfgs.depthvel_gridsearch_wf_plot(lithic_smallstack_streams, prior=[3630, 477])
    # print('Time taken: ' + str(time.time() - runtime_Start) + ' seconds')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()