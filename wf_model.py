import numpy as np

# Material vector: [rho, vp, vs]
# calculate the normal incidence reflectivity using density and velocity (aka impedance)
def normal_reflectivity(mat1, mat2):
    z1 = mat1[0] * mat1[1]
    z2 = mat2[0] * mat2[1]
    return (z2 - z1) / (z2 + z1)

def shuey_reflectivity(mat1, mat2, theta):
    # calculate the normal incidence reflectivity
    r0 = normal_reflectivity(mat1, mat2)

    deltavp = mat2[1] - mat1[1]
    deltavs = mat2[2] - mat1[2]
    deltarho = mat2[0] - mat1[0]
    # calculate the angle dependent reflectivity
    G = 0.5 * deltavp/mat1[1] - 2*(mat1[2]/mat1[1])**2 * (deltarho/mat1[0] + 2*deltavs/mat1[2])

    r = r0 + G * np.sin(theta)**2

    return r


def ricker_wavelet(frequency, duration, dt):
    t = np.arange(-duration/2, duration/2, dt)
    y = (1 - 2 * (np.pi**2) * (frequency**2) * (t**2)) * np.exp(-(np.pi**2) * (frequency**2) * (t**2))
    return t, y


def snell(theta1, mat1, mat2):
    vp1 = mat1[1]
    vp2 = mat2[1]
    return np.arcsin(vp1/vp2 * np.sin(theta1))


def compute_seismogram_2layer(t, source_wavelet, depth_interface, mat1, mat2, offset, dt):
    # Calculate the angle of incidence
    theta = np.arctan(offset / depth_interface)
    # Calculate reflection coefficient
    R = shuey_reflectivity(mat1, mat2, theta)

    # Compute time of reflection
    time_reflection = 2 * depth_interface / mat1[1] * np.cos(theta)
    reflection_arrival = np.zeros_like(t)
    reflection_arrival[int(time_reflection / dt)] = R

    # Convolve source with reflection
    seismogram = np.convolve(source_wavelet, reflection_arrival, mode='same')

    return seismogram


