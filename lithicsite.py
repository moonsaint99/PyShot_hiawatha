import numpy as np
import matplotlib as mpl
# mpl.use('macosx')
import matplotlib.pyplot as plt
import scipy as sp
import pickle as pkl
import op_pickfile
import pick_gridsearch as gs
import os

import su_pickfile as supf
import op_pickfile as pf
import pickanalysis as pa
import firn_analysis as fa

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


lithic_smallstack_names, primary_lithic_smallstacks, secondary_lithic_smallstacks, firn_lithic_smallstacks = op_pickfile.assimilate_pickdata('pickdata_lithic_smallstack')

primary_lithicstack = pf.opPickfile('primary_shotpicks_output/stack_onspread_good.csv')
secondary_lithicstack = pf.opPickfile('secondary_shotpicks_output/stack_onspread_good.csv')
primary_lithic_bigstack = [primary_lithicstack]
secondary_lithic_bigstack = [secondary_lithicstack]


# Perform a firn velocity inversion
inversion_results = fa.double_linear_exponential_fit(primary_lithic_smallstacks)

# Compute reflectivity and error reflectivities
ref_shots_simple_smallstack = []
ref_shots_simple_smallstack_upper_error = []
ref_shots_simple_smallstack_lower_error = []
for i in range(len(primary_lithic_smallstacks)):
    simple_reflectivity = pa.reflectivity(primary_lithic_smallstacks[i], secondary_lithic_smallstacks[i], 'simple', inversion_results, attn_coeff=2.7e-4, polarity='min')
    ref_shots_simple_smallstack.append(simple_reflectivity[0])
    ref_shots_simple_smallstack_upper_error.append(simple_reflectivity[1] - simple_reflectivity[0])
    ref_shots_simple_smallstack_lower_error.append(simple_reflectivity[0] - simple_reflectivity[2])
#
#     angle = np.rad2deg(primary_lithic_smallstacks[i].angle)
#     refl = ref_shots_simple_smallstack[i]
#     plt.plot(angle,refl, zorder=1, marker='o', markersize=5, linestyle='none')
#     plt.errorbar(np.rad2deg(primary_lithic_smallstacks[i].angle), ref_shots_simple_smallstack[i], yerr=[ref_shots_simple_smallstack_lower_error[i], ref_shots_simple_smallstack_upper_error[i]], zorder=0, fmt='none', ecolor='k', capsize=3)

ref_shots_simple_bigstack = []
ref_shots_simple_bigstack_upper_error = []
ref_shots_simple_bigstack_lower_error = []
reflectivity_fulldata = []
for i in range(len(primary_lithic_bigstack)):
    simple_reflectivity = pa.reflectivity(primary_lithic_bigstack[i], secondary_lithic_bigstack[i], 'simple', inversion_results, attn_coeff=2.7e-4, polarity='min')
    ref_shots_simple_bigstack.append(simple_reflectivity[0])
    ref_shots_simple_bigstack_upper_error.append(simple_reflectivity[1] - simple_reflectivity[0])
    ref_shots_simple_bigstack_lower_error.append(simple_reflectivity[0] - simple_reflectivity[2])

    angle = np.rad2deg(primary_lithic_bigstack[i].angle)
    refl = ref_shots_simple_bigstack[i]
    plt.plot(angle,refl, zorder=1, marker='o', markersize=5, linestyle='none')
    plt.errorbar(np.rad2deg(primary_lithic_bigstack[i].angle), ref_shots_simple_bigstack[i], yerr=[ref_shots_simple_bigstack_lower_error[i], ref_shots_simple_bigstack_upper_error[i]], zorder=0, fmt='none', ecolor='k', capsize=3)
    reflectivity_fulldata.append([primary_lithic_bigstack[i].angle, ref_shots_simple_bigstack[i], ref_shots_simple_bigstack_lower_error[i], ref_shots_simple_bigstack_upper_error[i]])
plt.grid()
plt.ylim([-0.2, 1])
plt.title('Reflectivity at lithic motion site')
plt.ylabel('Reflectivity')
plt.xlabel('Angle (deg)')
plt.show()

# Save the reflecitivity results
with open('reflectivity_lithic.pkl', 'wb') as f:
    pkl.dump(reflectivity_fulldata, f)


# gs.depthvel_gridsearch_plot([primary_lithic_fullstack], [secondary_lithic_fullstack], prior=[3710, 477])
