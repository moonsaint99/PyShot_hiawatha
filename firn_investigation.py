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
import pickle as pkl

# lithic_names, lithic_firn, lithic_secondary, lithic_firn_dummy = pf.assimilate_pickdata('pickdata_lithic_smallstack')
#
# # Perform a firn velocity and density inversion:
# inversion_results = fa.double_linear_exponential_fit(lithic_firn)
# lithic_firn_depth, lithic_firn_velocity = fa.firn_depth_vs_velocity(np.arange(0, 200, 1), inversion_results[0])
# plt.plot(lithic_firn_depth, lithic_firn_velocity, label='Firn velocity')
# plt.show()
# # Save the lithic site firn inversion results to a pkl file
# with open('lithic_firn_inversion_results.pkl', 'wb') as f:
#     pkl.dump(inversion_results, f)



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
# for shot in lithic_firn:
#     plt.scatter(shot.dist, shot.amplitude/shot.dist)
# plt.yscale('log')
# plt.ylim(1e-1, 1e3)
# plt.title('Geometric-spreading-corrected direct arrival amplitudes')
# plt.xlabel('Offset (m)')
# plt.ylabel('Amplitude')
# plt.show()

# # Assimilate all data into lists of distances and amplitudes
# lithic_firn_dist = []
# lithic_firn_amplitude = []
# for shot in lithic_firn:
#     lithic_firn_dist += np.ndarray.tolist(shot.dist)
#     inc_angle = fa.inc_angle(shot, inversion_results[0])
#     lithic_firn_amplitude += np.ndarray.tolist(shot.amplitude*np.cos(inc_angle)/shot.dist)
# lithic_firn_dist = np.array(lithic_firn_dist)
# # Use a linear regression to find the slope of the amplitude falloff
# lithic_firn_logamplitude = np.log(np.abs(lithic_firn_amplitude))
# slope, intercept, r_value, p_value, std_err = sp.stats.linregress(lithic_firn_dist, lithic_firn_logamplitude)
#
#
#
# # Correct amplitude falloff for an exponential attenuation with distance
# for shot in lithic_firn:
#     inc_angle = fa.inc_angle(shot, inversion_results[0])
#     amplitude_geomcorr = shot.amplitude * np.cos(inc_angle) / shot.dist
#     logamp = np.log(np.abs(amplitude_geomcorr))
#     plt.scatter(shot.dist, logamp)
# plt.plot(np.arange(0, 100, 1), slope*np.arange(0, 100, 1) + intercept, color='red')
# plt.title('Log geometric-corrected amplitude vs distance')
# plt.xlabel('Offset (m)')
# plt.ylabel('Log amplitude')
# plt.show()
#     plt.scatter(shot.dist, amplitude_corrected)
# plt.yscale('log')
# plt.ylim(1e-1, 1e3)
# plt.show()


# Assimilate aquifer site data
aquifer_names, aquifer_primary, aquifer_secondary, aquifer_firn = pf.assimilate_pickdata('aquifer_shots_picks')

inversion_results = fa.double_linear_exponential_fit(aquifer_firn)
aquifer_firn_depth, aquifer_firn_velocity = fa.firn_depth_vs_velocity(np.arange(0, 200, 1), inversion_results[0])
plt.plot(aquifer_firn_depth, aquifer_firn_velocity, label='Firn velocity')
plt.title('Aquifer Site Firn Velocity')
plt.show()
# Save the aquifer site firn inversion results to a pkl file
with open('aquifer_firn_inversion_results.pkl', 'wb') as f:
    pkl.dump(inversion_results, f)


# Visually validate goodness-of-fit of double-exponential fitted to direct arrivals
# for shot in aquifer_firn:
#     plt.plot(shot.dist, shot.time, linewidth=1, color='gray')
# theoretical_traveltime = fa.double_linear_exponential(np.arange(0, 200, 1), *inversion_results[0])
# plt.plot(np.arange(0, 200, 1), theoretical_traveltime, label='Theoretical traveltime', zorder=4, color='fuchsia', linewidth=3)
# plt.gca().invert_yaxis()
# plt.title('Firn picks')
# plt.xlabel('Offset (m)')
# plt.ylabel('Time (s)')
# plt.show()



# Assimilate aquifer stack data:
aquifer_stack_names, aquifer_stack_primary, aquifer_stack_secondary, aquifer_stack_firn = pf.assimilate_pickdata('aquifer_shotstack_picks')

# Assimilate all data into lists of distances and amplitudes
aquifer_stack_dist = []
aquifer_stack_amplitude = []
for shot in aquifer_stack_primary:
    aquifer_stack_dist += np.ndarray.tolist(shot.dist)
    inc_angle = fa.inc_angle(shot, inversion_results[0])
    aquifer_stack_amplitude += np.ndarray.tolist(shot.amplitude*np.cos(inc_angle)/shot.dist)
aquifer_stack_dist = np.array(aquifer_stack_dist)
# Use a linear regression to find the slope of the amplitude falloff
aquifer_stack_logamplitude = np.log(np.abs(aquifer_stack_amplitude))
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(aquifer_stack_dist, aquifer_stack_logamplitude)


# Correct amplitude falloff for an exponential attenuation with distance
for shot in aquifer_stack_primary:
    inc_angle = fa.inc_angle(shot, inversion_results[0])
    amplitude_geomcorr = shot.amplitude * np.cos(inc_angle) / shot.dist
    logamp = np.log(np.abs(amplitude_geomcorr))
    plt.scatter(shot.dist, logamp)
plt.plot(np.arange(0, 200, 1), slope*np.arange(0, 200, 1) + intercept, color='red')
# plt.ylim(1e-1, 1e3)
plt.show()
