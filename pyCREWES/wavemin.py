import numpy as np
import tntamp as tnt
import levrec as lev
import wavenorm as wn

def wavemin(dt, fdom=15.0, tlength=None, m=4, stab=0.000001):

    if m < 2 or m > 7:
        raise ValueError("m must lie between 2 and 7")

    # adjust fdom to account for the weirdness of tntamp
    if m == 2:
        f0, m0 = -0.0731, 1.0735
    elif m == 3:
        f0, m0 = 0.0163, 0.9083
    elif m == 4:
        f0, m0 = 0.0408, 0.8470
    elif m == 5:
        f0, m0 = -0.0382, 0.8282
    elif m == 6:
        f0, m0 = 0.0243, 0.8206
    elif m == 7:
        f0, m0 = 0.0243, 0.8206
    fdom2 = (fdom - f0) / m0

    if tlength is None:
        tlength = 127.0 * dt

    # create a time vector
    nt = int(round(2.0 * tlength / dt) + 1)
    nt = 2 ** int(np.ceil(np.log2(nt)))
    tmax = dt * (nt - 1)
    tw = np.arange(0.0, tmax + dt, dt)

    # create a frequency vector
    fnyq = 1.0 / (2.0 * tw[1] - tw[0])
    f = np.linspace(0.0, fnyq, len(tw) // 2 + 1)

    # create the power spectrum
    tmp = tnt.tntamp(fdom2, f, m)  # Assuming tntamp is a function you have elsewhere
    powspec = np.square(tmp)

    # create the autocorrelation
    auto = np.fft.irfft(powspec)
    auto[0] = auto[0] * (1 + stab)

    # run this through Levinson
    nlags = int(round(tlength / dt) + 1)
    b = np.zeros(nlags)
    b[0] = 1.0
    winv = lev.levrec(auto[:nlags], b)  # Assuming levrec is a function you have elsewhere

    # invert the winv
    w = np.real(np.fft.ifft(1.0 / np.fft.fft(winv)))

    # now normalize the w
    w = wavenorm(w, tw, 2)  # Assuming wavenorm is a function you have elsewhere

    return w, tw
