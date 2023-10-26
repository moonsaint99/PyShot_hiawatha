import numpy as np
import wf_model as wfm
import matplotlib.pyplot as plt
import obspy as op
import time
import multiprocessing
from collections import deque


def l2_misfit(streamdict, current_model):
    l2 = 0
    trlength = 0
    npts = 0
    for filename, stream in streamdict.items():
        for trace in stream:
            current_synthetic = current_model.get_synthetic(np.arange(0, trace.stats.npts*trace.stats.delta,
                                                                      trace.stats.delta), trace.stats.distance)
            l2 += np.linalg.norm(trace.data - current_synthetic)**2
            if trlength == 0:
                trlength = len(trace.data)
                npts = trace.stats.npts
            else:
                pass
    # Normalize by number of traces
    # l2 /= len(streamdict)*trlength
    l2 = np.sqrt(l2)/npts/len(streamdict)/trlength
    return l2


def metropolis_hastings(initial_params, iterations, streamdict, proposal_std, step_size=1):
    current_params = initial_params
    proposed_params = initial_params
    current_model = wfm.Model(current_params[0], current_params[1], current_params[2], current_params[3])
    current_l2 = l2_misfit(streamdict, current_model)

    accepted_params = []
    acceptance_record = deque(maxlen=100)
    for _ in range(iterations):
        # Propose a new model
        for i in range(len(current_params)):
            for j in range(len(current_params[i])):
                proposed_params[i][j] = current_params[i][j] + proposal_std[i][j] * np.random.randn()

        # Generate new synthetic seismogram and compute L2 distance
        proposed_model = wfm.Model(proposed_params[0], proposed_params[1], proposed_params[2], proposed_params[3])
        proposed_l2 = l2_misfit(streamdict, proposed_model)

        rolling_acceptance_ratio = sum(acceptance_record) / (len(acceptance_record)+1)

        if rolling_acceptance_ratio < 0.3:
            step_size *= 1.0001
            if rolling_acceptance_ratio < 0.2:
                step_size *= 1.001
        elif rolling_acceptance_ratio > 0.4:
            step_size *= 0.9999
            if rolling_acceptance_ratio > 0.5:
                step_size *= 0.999

        if _%25 == 0:
            print('Acceptance ratio is ' + str(rolling_acceptance_ratio))

        if _%500 == 0:
            for stream in streamdict.values():
                for trace in stream:
                    if trace.stats.distance == 50:
                        plt.plot(np.arange(0, trace.stats.npts*trace.stats.delta, trace.stats.delta), trace.data/np.max(trace.data))
                        plt.plot(np.arange(0, trace.stats.npts*trace.stats.delta, trace.stats.delta),
                                 proposed_model.get_synthetic(np.arange(0, trace.stats.npts*trace.stats.delta,
                                                                        trace.stats.delta), trace.stats.distance))
                        plt.show()


        # Metropolis criterion
        if np.random.rand() < np.exp((current_l2 - proposed_l2) / step_size):
            print(_)
            print(current_l2)
            print(proposed_l2)
            print(current_params)
            current_params = proposed_params
            current_l2 = proposed_l2
            acceptance_record.append(1)
            # # plot the synthetic seismogram
            # plt.plot(np.arange(0, 0.3, 0.00025), proposed_model.get_synthetic(np.arange(0, 0.3, 0.00025), 50))
            # plt.show()
            accepted_params.append(current_params)
        else:
            acceptance_record.append(0)

    return accepted_params