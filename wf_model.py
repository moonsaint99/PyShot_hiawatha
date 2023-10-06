import numpy as np
import os
import scipy as sp
import zoeppritz as zp

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


def compute_seismogram_2layer(t, depth_interface, mat1, mat2, offset, source_freq=125, source_duration=0.025, dt=0.00025):
    source_wavelet = ricker_wavelet(source_freq, source_duration, dt)[1]
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


def retrieve_seismogram_2layer(t, depth_interface, mat1, mat2, offset, source_freq=125, source_duration=0.025,
                               dt=0.00025):
    parameters = (t, depth_interface, mat1, mat2, offset, source_freq, source_duration, dt)
    filename = "_".join(str(p) for p in parameters) + ".npy"

    if os.path.exists('./model_cache/' + filename):
        seismogram = np.load('./model_cache/' + filename)
    else:
        seismogram = compute_seismogram_2layer(*parameters)
        np.save('./model_cache/' + filename, seismogram)

    return seismogram


# def compute_seismogram_3layer(t, source_wavelet, depth_1, depth_2, mat1, mat2, mat3, offset, dt):
#     pass


class Model:
    def __init__(self, interface_depths, rho, vp, vs):
        self.interface_depths = interface_depths
        self.rho = rho
        self.vp = vp
        self.vs = vs

    def mat(self, n):
        return [self.rho[n], self.vp[n], self.vs[n]]

    # def normal_reflectivity(self):
    #     reflectivity = []
    #     for n in range(len(self.interface_depths)):
    #         reflectivity.append(normal_reflectivity(self.mat(n), self.mat(n+1)))
    #     return reflectivity

    # def timing(self, offset):
    #     timing = []
    #     for n in range(len(self.interface_depths)):

    def angle_optimalness(self, theta, interface, offset):  # Used to determine takeoff angle for the reflection off each interface
        output = self.interface_depths[0] * np.tan(theta) - offset / 2
        for i in range(len(self.interface_depths[1:interface])):
            output += self.interface_depths[i + 1] * np.tan(
                np.arcsin(self.vp[i + 1] / self.vp[0] * np.sin(theta)))
        return output ** 2

    def gen_synthetic(self, t, offset, source_freq=125, source_duration=0.025, dt=0.00025):
        source_wavelet = ricker_wavelet(source_freq, source_duration, dt)[1]
        # We estimate the takeoff angle using scipy minimize
        depth = 0
        theta_takeoff = [] # Takeoff angle for the ray that reflects off interface i
        for i in range(len(self.interface_depths)):
            depth += self.interface_depths[i]
            theta_takeoff.append(sp.optimize.minimize(self.angle_optimalness, x0=np.arctan((offset/2)/depth), args=(i,offset),
                                              method='Powell').x[0])#, bounds=[(0, np.deg2rad(30))]).x[0])  # Powell method
                                                                            # seems to be the fastest for this case
        print(np.rad2deg(theta_takeoff))

        def theta(i,j):
            # i is the reflector index, j is the interface index
            return np.arcsin(self.vp[j]/self.vp[0]*np.sin(theta_takeoff[i]))

        # Calculate reflection and transmission coefficient at each interface, given known takeoff angle
        R = []
        T = []
        for i in range(len(self.interface_depths)):
            incidence_angle = theta(i,i)
            # R.append(shuey_reflectivity(self.mat(i), self.mat(i+1), incidence_angle))
            ref = zp.ref_trans_array(self.mat(i), self.mat(i + 1), incidence_angle)
            Rnew, junk1, Tnew, junk2 = zp.ref_trans_array(self.mat(i), self.mat(i+1), incidence_angle)
            R.append(Rnew)
            T.append(Tnew)

        # Calculate relative arrival amplitudes
        A = R
        for i in range(len(R)):
            for j in range(i):
                A[i] *= T[j]**2

        # Calculate arrival timing
        time_reflection = []
        for i in range(len(self.interface_depths)):  # i is the reflection index, i=0 for first reflection, i=1 for 2nd
            time_reflection.append(0)
            for j in range(i+1):  # j is an index to iterate through all layers above the reflection.
                time_reflection[i] += (2 * self.interface_depths[j] / self.vp[j] / np.cos(
                    theta(i,i)))
        # Calculate impulse response
        greens = np.zeros_like(t)
        for i in range(len(R)):
            greens[int(time_reflection[i] / dt)] = A[i]

        # Convolve source with reflection
        seismogram = np.convolve(source_wavelet, greens, mode='same')

        # for i in range(len(self.interface_depths) - 1):
        #     seismogram += retrieve_seismogram_2layer(t, self.interface_depths[i], [self.rho[i], self.vp[i], self.vs[i]],
        #                                              [self.rho[i+1], self.vp[i+1], self.vs[i+1]], offset,
        #                                              source_freq=source_freq, source_duration=source_duration, dt=dt)
        return seismogram

    def get_synthetic(self, t, offset, source_freq=125, source_duration=0.025, dt=0.00025):
        parameters = self.interface_depths, self.rho, self.vp, self.vs, t, offset, source_freq, source_duration, dt
        filename = "_".join(str(p) for p in parameters) + ".npy"
        filename = str(hash(filename))
        if os.path.exists('./model_cache/' + filename):
            seismogram = np.load('./model_cache/' + filename)
        else:
            seismogram = self.gen_synthetic(t, offset, source_freq, source_duration, dt)
            np.save('./model_cache/' + filename, seismogram)

        return seismogram