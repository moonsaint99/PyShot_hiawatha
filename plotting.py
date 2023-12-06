import numpy as np
import matplotlib as mpl
mpl.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import scipy as sp
import pickle as pkl
import zoeppritz as zp

cm = 1 / 2.54
mm = 1 / 25.4


def firndensity_robin(velocity):
    return 0.221*velocity+59

def firndensity_kohnnen(velocity, icevelocity):
    return 915/(1+((icevelocity-velocity)/2250)**1.22)

if False:
    # Plot firn velocity profiles at the two sites
    # Load the saved depth and velocity arrays
    aquifer_firn = np.load('tmp/depthvel_aquifer.npy')
    lithic_firn = np.load('tmp/depthvel_lithic.npy')
    # Plot the velocity profiles
    plt.figure(figsize=(95*mm, 115*mm))
    plt.plot(lithic_firn[1], lithic_firn[0], 'r', label='Site 1 velocity')
    plt.plot(aquifer_firn[1], aquifer_firn[0], 'b', label='Site 2 velocity')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.title('Firn velocity profiles')
    plt.legend()
    plt.grid()
    # plt.grid(which='minor', linestyle='--')
    plt.minorticks_on()
    plt.ylim(0, 30)
    plt.gca().invert_yaxis()
    plt.savefig('tmp/depthvel_firn.png')
    plt.show()

if False:
    # Plot firn velocity profiles
    # Using a second axis, plot firn density profiles
    # Load the saved depth and velocity arrays
    aquifer_firn = np.load('tmp/depthvel_aquifer.npy')
    lithic_firn = np.load('tmp/depthvel_lithic.npy')
    # Plot the velocity profiles
    fig, ax1 = plt.subplots(figsize=(95*mm, 115*mm))
    # plt.figure(figsize=(95*mm, 115*mm))
    # ax1.plot(lithic_firn[1], lithic_firn[0], 'r', label='Site 1 velocity')
    # ax1.plot(aquifer_firn[1], aquifer_firn[0], 'b', label='Site 2 velocity')
    # ax1.set_xlabel('Density (kg/m³), Velocity (m/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Density (kg/m³)')
    # ax1.set_title('Firn density and velocity profiles')
    ax1.grid()
    # ax1.grid(which='minor', linestyle='--')
    ax1.minorticks_on()
    # ax1.set_ylim(0, 40)
    # ax1.invert_yaxis()
    # Plot the density profiles
    # ax2 = ax1.twinx()
    ax1.plot(firndensity_kohnnen(lithic_firn[1], np.nanmax(lithic_firn)), lithic_firn[0], 'r--', label='Site 1 Kohnen density')
    ax1.plot(firndensity_kohnnen(aquifer_firn[1], np.nanmax(aquifer_firn)), aquifer_firn[0], 'b--', label='Site 2 Kohnen density')
    ax1.plot(firndensity_robin(lithic_firn[1]), lithic_firn[0], 'r', label='Site 1 Robin density')
    ax1.plot(firndensity_robin(aquifer_firn[1]), aquifer_firn[0], 'b', label='Site 2 Robin density')
    ax1.set_xlabel('Density (kg/m³)')
    ax1.set_ylim(0, 30)
    ax1.invert_yaxis()
    # Retrieve handles and labels for ax1 and ax2
    handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax1.get_legend_handles_labels()

    # Combine handles and labels
    # all_handles = handles1 + handles2
    # all_labels = labels1 + labels2

    # Create a single legend for the combined handles and labels
    ax1.legend(handles1, labels1, loc='lower left')
    # ax2.set_yticklabels([])
    plt.savefig('tmp/depthvel_firn_density.png')
    plt.show()

if True:

    plt.figure(figsize=(190*mm, 115*mm))
    # Load reflectivities from files
    with open('reflectivity_aquifer.pkl', 'rb') as f:
        reflectivity_aquifer = pkl.load(f)
    with open('reflectivity_lithic.pkl', 'rb') as f:
        reflectivity_lithic = pkl.load(f)
    # Plot reflectivities
    plt.plot(np.rad2deg(reflectivity_lithic[0][0]), reflectivity_lithic[0][1], 'r', label='Site 1 reflectivity', linestyle='none', marker='o', markersize=5, zorder=5)
    plt.errorbar(np.rad2deg(reflectivity_lithic[0][0]), reflectivity_lithic[0][1], yerr=[reflectivity_lithic[0][2], reflectivity_lithic[0][3]], zorder=5, fmt='none', ecolor='r', capsize=3)
    plt.plot(np.rad2deg(reflectivity_aquifer[0][0]), reflectivity_aquifer[0][1], 'b', label='Site 2 reflectivity', linestyle='none', marker='o', markersize=5, zorder=4)
    plt.errorbar(np.rad2deg(reflectivity_aquifer[0][0]), reflectivity_aquifer[0][1], yerr=[reflectivity_aquifer[0][2], reflectivity_aquifer[0][3]], zorder=4, fmt='none', ecolor='b', capsize=3)
    plt.xlabel('Angle of Incidence (º)')
    plt.ylabel('Basal Reflectivity')
    plt.ylim([-1, 0.75])
    plt.xlim([0, 12])
    plt.grid()

    theta_range = np.deg2rad(np.arange(0, 12, 0.1))

    consolidated_sediment_min = [1600, 2000, 1000]
    consolidated_sediment_max = [1900, 2600, 1400]
    consolidated_sediment_window = zp.reflectivity_window(consolidated_sediment_min, consolidated_sediment_max, theta_range)
    plt.plot(np.rad2deg(theta_range), consolidated_sediment_window[0], 'r', label='Consolidated sediment max')
    plt.plot(np.rad2deg(theta_range), consolidated_sediment_window[1], 'r', label='Consolidated sediment min')
    plt.fill_between(np.rad2deg(theta_range), consolidated_sediment_window[0], consolidated_sediment_window[1], color='r', alpha=0.3)

    unconsolidated_sediment_min = [1600, 1700, 900]
    unconsolidated_sediment_max = [1800, 1900, 1200]
    unconsolidated_sediment_window = zp.reflectivity_window(unconsolidated_sediment_min, unconsolidated_sediment_max, theta_range)
    plt.plot(np.rad2deg(theta_range), unconsolidated_sediment_window[0], 'b', label='Unconsolidated sediment max')
    plt.plot(np.rad2deg(theta_range), unconsolidated_sediment_window[1], 'b', label='Unconsolidated sediment min')
    plt.fill_between(np.rad2deg(theta_range), unconsolidated_sediment_window[0], unconsolidated_sediment_window[1], color='b', alpha=0.3)

    lithified_sediment_min = [2200, 3000, 1200]
    lithified_sediment_max = [2450, 3750, 2450]
    lithified_sediment_window = zp.reflectivity_window(lithified_sediment_min, lithified_sediment_max, theta_range)
    plt.plot(np.rad2deg(theta_range), lithified_sediment_window[0], 'y', label='Lithified sediment max')
    plt.plot(np.rad2deg(theta_range), lithified_sediment_window[1], 'y', label='Lithified sediment min')
    plt.fill_between(np.rad2deg(theta_range), lithified_sediment_window[0], lithified_sediment_window[1], color='y', alpha=0.3)


    bedrock_min = [2700, 5200, 2700]
    bedrock_max = [2800, 6200, 3400]
    bedrock_window = zp.reflectivity_window(bedrock_min, bedrock_max, theta_range)
    plt.plot(np.rad2deg(theta_range), bedrock_window[0], 'gray', label='Bedrock max')
    plt.plot(np.rad2deg(theta_range), bedrock_window[1], 'gray', label='Bedrock min')
    plt.fill_between(np.rad2deg(theta_range), bedrock_window[0], bedrock_window[1], color='gray', alpha=0.3)

    water = [1000, 1498, 0]
    water_array = zp.reflectivity_array(water, theta_range)
    plt.plot(np.rad2deg(theta_range), water_array, 'k', label='Water')

    # Legend Definition
    # Patch objects:
    patch_bedrock = mpatches.Patch(color='gray', label='Bedrock')
    patch_lithified_sediment = mpatches.Patch(color='y', label='Lithified sediment')
    patch_unconsolidated_sediment = mpatches.Patch(color='b', label='Unconsolidated sediment')
    patch_consolidated_sediment = mpatches.Patch(color='r', label='Consolidated sediment')

    # Line objects:
    line_water = mlines.Line2D([], [], color='k', label='Water')

    # Scatter objects
    scatter_site1 = plt.scatter([], [], color='r', label='Site 1', marker='o')
    scatter_site2 = plt.scatter([], [], color='b', label='Site 2', marker='o')

    # Legend Entries:
    legend_entries = [
        mlines.Line2D([], [], color='none', label='Calculated'),
        scatter_site1,
        scatter_site2,
        mlines.Line2D([], [], color='none', label=''),
        mlines.Line2D([], [], color='none', label=''),
        mlines.Line2D([], [], color='none', label=''),
        mlines.Line2D([], [], color='none', label='Modeled'),
        patch_bedrock,
        patch_lithified_sediment,
        patch_consolidated_sediment,
        patch_unconsolidated_sediment,
        line_water
    ]


    plt.legend(
        ncols=2,
        loc='lower right',
        handles=legend_entries,
        framealpha=1,
        edgecolor='black',
        facecolor='white',
        fontsize=8
    )
    plt.show()


