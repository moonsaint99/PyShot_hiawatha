import numpy as np
import matplotlib as mpl
# mpl.use('macosx')
import matplotlib.pyplot as plt
import scipy as sp
import firn_analysis as fa
import op_pickfile as pf
import pickanalysis as pa
import os
import csv
import time

lithic_names, lithic_firn, lithic_secondary, lithic_firn_dummy = pf.assimilate_pickdata('pickdata_lithic_smallstack')

# Perform a firn velocity and density inversion:
inversion_results = fa.double_linear_exponential_fit(lithic_firn)
lithic_firn_depth, lithic_firn_velocity = fa.firn_depth_vs_velocity(np.arange(0, 200, 1), inversion_results[0])
# plt.plot(lithic_firn_depth, lithic_firn_velocity, label='Firn velocity')
# plt.show()


# Visually validate goodness-of-fit of double-exponential fitted to direct arrivals
# for shot in lithic_firn:
#     plt.plot(shot.dist, shot.time, linewidth=1, color='gray')
# theoretical_traveltime = fa.double_linear_exponential(np.arange(0, 120, 1), *inversion_results[0])
# plt.plot(np.arange(0, 120, 1), theoretical_traveltime, label='Theoretical traveltime', zorder=4, color='fuchsia', linewidth=3)
# plt.gca().invert_yaxis()
# plt.title('Firn picks')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.show()

# Now we show amplitude falloff with distance:
for shot in lithic_firn:
    plt.scatter(shot.dist, shot.amplitude/shot.dist)
plt.yscale('log')
plt.ylim(1e-1, 1e3)
plt.title('Geometric-spreading-corrected direct arrival amplitudes')
plt.xlabel('Offset (m)')
plt.ylabel('Amplitude')
plt.show()

# Correct amplitude falloff for an exponential attenuation with distance
for shot in lithic_firn:
    attn_corr = np.exp(shot.dist * -4e-2)
    amplitude_corrected = shot.amplitude / attn_corr / shot.dist
    plt.scatter(shot.dist, amplitude_corrected)
plt.yscale('log')
plt.ylim(1e-1, 1e3)
plt.show()