import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def firndensity_robin(velocity):
    return 0.221*velocity+59

def firndensity_kohnnen(velocity, icevelocity):
    return 915/(1+((icevelocity-velocity)/2250)**1.22)

if True:
    # Plot firn velocity profiles at the two sites
    # Load the saved depth and velocity arrays
    aquifer_firn = np.load('tmp/depthvel_aquifer.npy')
    lithic_firn = np.load('tmp/depthvel_lithic.npy')
    # Plot the velocity profiles
    plt.plot(lithic_firn[1], lithic_firn[0], 'r', label='Site 1 velocity')
    plt.plot(aquifer_firn[1], aquifer_firn[0], 'b', label='Site 2 velocity')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Depth (m)')
    plt.title('Firn velocity profiles')
    plt.legend()
    plt.grid()
    # plt.grid(which='minor', linestyle='--')
    plt.minorticks_on()
    plt.ylim(0, 40)
    plt.gca().invert_yaxis()
    plt.savefig('tmp/depthvel_firn.png')
    plt.show()

if True:
    # Plot firn velocity profiles
    # Using a second axis, plot firn density profiles
    # Load the saved depth and velocity arrays
    aquifer_firn = np.load('tmp/depthvel_aquifer.npy')
    lithic_firn = np.load('tmp/depthvel_lithic.npy')
    # Plot the velocity profiles
    fig, ax1 = plt.subplots()
    ax1.plot(lithic_firn[1], lithic_firn[0], 'r', label='Site 1 velocity')
    ax1.plot(aquifer_firn[1], aquifer_firn[0], 'b', label='Site 2 velocity')
    ax1.set_xlabel('Density (kg/m³), Velocity (m/s)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Firn density and velocity profiles')
    ax1.grid()
    # ax1.grid(which='minor', linestyle='--')
    ax1.minorticks_on()
    ax1.set_ylim(0, 40)
    ax1.invert_yaxis()
    # Plot the density profiles
    ax2 = ax1.twinx()
    ax2.plot(firndensity_kohnnen(lithic_firn[1], np.nanmax(lithic_firn)), lithic_firn[0], 'r--', label='Site 1 Kohnen density')
    ax2.plot(firndensity_kohnnen(aquifer_firn[1], np.nanmax(aquifer_firn)), aquifer_firn[0], 'b--', label='Site 2 Kohnen density')
    ax2.plot(firndensity_robin(lithic_firn[1]), lithic_firn[0], 'r:', label='Site 1 Robin density')
    ax2.plot(firndensity_robin(aquifer_firn[1]), aquifer_firn[0], 'b:', label='Site 2 Robin density')
    ax2.set_xlabel('Density (kg/m³)')
    ax2.set_ylim(0, 40)
    ax2.invert_yaxis()
    # Retrieve handles and labels for ax1 and ax2
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine handles and labels
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Create a single legend for the combined handles and labels
    ax1.legend(all_handles, all_labels, loc='lower center')
    ax2.set_yticklabels([])
    plt.savefig('tmp/depthvel_firn_density.png')
    plt.show()