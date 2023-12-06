import numpy as np
import scipy as sp
import firn_analysis as fa

def attenuation(x, a, b):
    return b*np.exp(-a*x)


def attn_fit(args):
    dist_array = np.array([])
    min_array = np.array([])
    for i in args:
        dist_array = np.append(dist_array, i.dist_no_outliers)
        newmin = i.min_no_outliers/np.sqrt(i.dist_no_outliers)
        min_array = np.append(min_array, newmin)
    return sp.optimize.curve_fit(
        f=attenuation,
        xdata=dist_array,
        ydata=min_array,
        p0=np.array([0.02, 2.1e2])
    )


# def primary_amp(primary, attn_coeff, inc_angle):
#     geom_corr = np.cos(inc_angle)/primary.dist
#     attn_corr = np.exp(-attn_coeff*primary.dist)
#     return primary.min/attn_corr/geom_corr*100


# def reflectivity(primary, secondary, attn_coeff, polarity='max'):
#     path_length = 2*np.sqrt(primary.dist**2 + 477**2)
#     geom_corr = np.cos(primary.angle)/path_length
#     attn_corr = np.exp(-attn_coeff*primary.dist)
#     if polarity == 'max':
#         return secondary.max/primary_amp(primary, attn_coeff, inc_angle(primary, inversion_results[0]))/attn_corr/geom_corr
#     elif polarity == 'min':
#         return secondary.min/primary_amp(primary, attn_coeff, inc_angle(primary, inversion_results[0]))/attn_corr/geom_corr


def pair_finder(primary):
    # This function will take a pickfile of primary arrivals and return a list of
    # every primary arrival that has a complementary double pathlength arrival
    dist_tuple_list = []
    index_tuple_list = []
    for dist1 in primary.dist:
        for dist2 in primary.dist:
            if dist2 == dist1*2:
                dist_tuple_list.append((dist1, dist2))
                index_tuple_list.append((np.where(primary.dist == dist1)[0][0], np.where(primary.dist == dist2)[0][0]))
    return index_tuple_list, dist_tuple_list


def pair_source_amplitudes(primary):
    # Find the source amplitude given a primary pickfile
    # This function will take a pickfile of primary arrivals and return a list of
    # estimates of source amplitude from every pair of direct arrivals
    inversion_results = fa.double_linear_exponential_fit([primary])
    incidence_angle = fa.inc_angle(primary, inversion_results[0])

    index_tuple_list, dist_tuple_list = pair_finder(primary)
    if len(index_tuple_list) == 0:
        return 'No pairs found'
    source_amp_list = []
    for i in index_tuple_list:
        path_factor_0 = np.cos(incidence_angle[i[0]])/primary.dist[i[0]]
        path_factor_1 = np.cos(incidence_angle[i[1]])/primary.dist[i[1]]
        source_amp = np.abs(primary.min[i[0]])**2/np.abs(primary.min[i[1]]) * path_factor_1/(path_factor_0**2)
        source_amp_list.append(source_amp)
    return source_amp_list


def dir_lin_fwd_model(primary, attn_coeff, source_amplitude):
    # This function will take a pickfile of primary arrivals and return a list of
    # estimates of source amplitude from every pair of direct arrivals
    inversion_results = fa.double_linear_exponential_fit([primary])
    incidence_angle = fa.inc_angle(primary, inversion_results[0])
    path_factor = np.cos(incidence_angle)/primary.dist
    return source_amplitude * path_factor * np.exp(-attn_coeff*primary.dist)


def dir_lin_source_amplitudes(primary, inversion_results=0): # this should be restricting to greater amplitudes
    if inversion_results ==0:
        inversion_results = fa.double_linear_exponential_fit([primary])
    incidence_angle = fa.inc_angle(primary, inversion_results[0])
    path_factor = np.cos(incidence_angle)/primary.dist
    y = np.log(np.abs(primary.min/path_factor))
    x = primary.dist
    attenuation, logamplitude, r_value, p_value, std_err = sp.stats.linregress(x, y)
    return attenuation, logamplitude, r_value, p_value, std_err
    # Use scipy.optimize.curve_fit to fit dir_lin_fwd_model to the data
    # fit_results = sp.optimize.curve_fit(
    #     f=dir_lin_fwd_model,
    #     xdata=primary,
    #     ydata=-primary.min,
    #     p0=np.array([0.00025, 1000]),
    # )
    # return fit_results[0][0], fit_results[0][1]


def simple_source_amplitude(primary, attn_coeff, inv_results):
    incidence_angle = fa.inc_angle(primary, inv_results[0])
    path_factor = np.cos(incidence_angle)/primary.dist
    source_amplitude = np.abs(primary.amplitude)/path_factor/np.exp(attn_coeff*primary.dist)
    return source_amplitude


def reflectivity(primary, secondary, source_amplitude, inv_results, attn_coeff=2.7e-4, attn_firn=-5.18e-2, polarity='max',  attn_hi = 4e-4, attn_lo = 2e-4):
    # Source amplitude calibration
    dir_lin_amp_results = dir_lin_source_amplitudes(primary, inv_results)
    if attn_coeff == 0:
        attn_coeff = -dir_lin_amp_results[0]
    if source_amplitude == 'pair':
        pair_amplitude = pair_source_amplitudes(primary)
        source_type = 'pair'
        if pair_amplitude == 'No pairs found':
            return 'No pairs found'
        # pair_amplitude = pair_amplitude[1:]
        source_amplitude = np.mean(pair_amplitude)
    elif source_amplitude == 'dir_lin':
        source_type = 'dir_lin'
        # source_amplitude = np.exp(dir_lin_amp_results[1])
        source_amplitude = dir_lin_amp_results[1]
    elif source_amplitude == 'simple':
        source_type = 'simple'
        source_amplitude = (simple_source_amplitude(primary, attn_firn, inv_results))

    # Geometric correction
    path_length = 2*np.sqrt((primary.dist/2)**2 + 477**2)
    imp_contr = 1
    geom_corr = np.cos(0)/path_length*imp_contr

    # Attenuation correction
    attn_corr = np.exp(attn_coeff*path_length)
    attn_corr_hi = np.exp(attn_hi*path_length)
    attn_corr_lo = np.exp(attn_lo*path_length)

    print('Source amplitude: ' + str(source_amplitude))
    print('Geometric correction: ' + str(1/geom_corr))
    print('Attenuation correction: ' + str(attn_corr))

    refl = secondary.amplitude / np.abs(source_amplitude) * attn_corr / geom_corr
    refl_upper = secondary.deviation(1).amplitude / np.abs(source_amplitude) * attn_corr_lo / geom_corr
    refl_lower = secondary.deviation(-1).amplitude / np.abs(source_amplitude) * attn_corr_hi / geom_corr
    return refl, refl_upper, refl_lower


def refl_time(offset, angle, velocity, depth=405):
    refl_timing = 2*np.sqrt(
                            (depth ** 2) +
                            ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    ) / velocity
    return refl_timing