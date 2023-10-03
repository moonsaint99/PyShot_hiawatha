import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import gridsearch as gs

# This script will use picked amplitudes of the primary and secondary seismic
# arrivals to calculate the ratio of the two amplitudes.


pvel = 3800

import pickfile as pf

def attenuation(x, a, b):
    return b*np.exp(-a*x)


def attn_fit(args):
    dist_array = np.array([])
    min_array = np.array([])
    for i in args:
        dist_array = np.append(dist_array, i.dist)
        min_array = np.append(min_array, i.min)
    return sp.optimize.curve_fit(
        f=attenuation,
        xdata=dist_array,
        ydata=min_array,
        p0=np.array([0.02, 2.1e2])
    )


def inv1(x, a, b, c, d, f):
    return a*(1-np.exp(-b*x)) + c*(1-np.exp(-d*x)) + f*x


def inv1_fit(args):
    dist_array = np.array([])
    tmin_array = np.array([])
    for i in args:
        dist_array = np.append(dist_array, i.dist)
        tmin_array = np.append(tmin_array, i.tmin)
    return sp.optimize.curve_fit(
        f=inv1,
        xdata=dist_array,
        ydata=tmin_array,
        # p0=np.array([0.85/100,0.035,1/1000,1.4, 1/3900]),
        p0=([0.011,0.04, 0.35, 0.005, 1/3800]),
        maxfev=10000000
    )


def inv1_slope(x, params):  # This will accept a tuple of parameters
    a, b, c, d, f = params
    dtdx = a*b*np.exp(-b*x) + c*d*np.exp(-d*x) + f
    return 1/dtdx


def inv1_depth(dist, params):
    vel_grad = inv1_slope(dist, params)
    # vel_apparent = primary.dist_no_outliers/primary.tmin_no_outliers
    vel_apparent = dist/inv1(dist, *params)
    z = np.array([])
    for i in range(len(dist)):
        z_int = sp.integrate.quad(lambda x: 1/(np.arccosh(vel_grad[i]/vel_apparent[i])), 0, dist[i])
        z_temp = 1/np.pi * z_int[0]
        z = np.append(z, z_temp)
    return z


def inc_angle(primary, params):
    ray_param = 1/inv1_slope(primary.dist, params)
    return np.arcsin(ray_param*1000)

primary_29 = pf.Pickfile('pickdata_lithic/29_primary.info', 'incr', 115, outliers=3)
primary_30 = pf.Pickfile('pickdata_lithic/30_primary.info', 'incr', 115, outliers=3)
primary_31 = pf.Pickfile('pickdata_lithic/31_primary.info', 'incr', 115, outliers=3)
primary_33 = pf.Pickfile('pickdata_lithic/33_primary.info', 'incr', 100, outliers=5)
primary_34 = pf.Pickfile('pickdata_lithic/34_primary.info', 'incr', 100, outliers=5)

secondary_29 = pf.Pickfile('pickdata_lithic/29_secondary.info', 'incr', 'secondary', outliers=3)
secondary_30 = pf.Pickfile('pickdata_lithic/30_secondary.info', 'incr', 'secondary', outliers=3)
secondary_31 = pf.Pickfile('pickdata_lithic/31_secondary.info', 'incr', 'secondary', outliers=3)
secondary_33 = pf.Pickfile('pickdata_lithic/33_secondary.info', 'incr', 'secondary', outliers=5)
secondary_34 = pf.Pickfile('pickdata_lithic/34_secondary.info', 'incr', 'secondary', outliers=5)

primary_72 = pf.Pickfile('pickdata_aquifer/72_wind_primary.info', 'incr', 195)
primary_73 = pf.Pickfile('pickdata_aquifer/73_wind_primary.info', 'incr', 195)
primary_74 = pf.Pickfile('pickdata_aquifer/74_wind_primary.info', 'incr', 195)
midsouth = 135
primary_76 = pf.Pickfile('pickdata_aquifer/76_wind_primary.info', 'incr', midsouth)
primary_77 = pf.Pickfile('pickdata_aquifer/77_wind_primary.info', 'incr', midsouth)
midspread = 135
primary_78 = pf.Pickfile('pickdata_aquifer/78_wind_primary.info', 'incr', midspread)
primary_79 = pf.Pickfile('pickdata_aquifer/79_wind_primary.info', 'incr', midspread)
primary_80 = pf.Pickfile('pickdata_aquifer/80_wind_primary.info', 'incr', 60, 3)
primary_81 = pf.Pickfile('pickdata_aquifer/81_wind_primary.info', 'incr', 60, 3)
primary_82 = pf.Pickfile('pickdata_aquifer/82_wind_primary.info', 'incr', 60, 3)

aquifer_primary_list = [primary_72, primary_73, primary_74, primary_76, primary_77, primary_78, primary_79, primary_80, primary_81, primary_82]

primary_lithic_fullstack = pf.Pickfile('pickdata_lithic/fullstack_stacked_primary.info', 'incr', 0, 3, maxrows=24)

secondary_74 = pf.Pickfile('pickdata_aquifer/74_wind_secondary.info', 'incr', 195)
secondary_73 = pf.Pickfile('pickdata_aquifer/73_wind_secondary.info', 'incr', 195)
secondary_72 = pf.Pickfile('pickdata_aquifer/72_wind_secondary.info', 'incr', 195)
secondary_76 = pf.Pickfile('pickdata_aquifer/76_wind_secondary.info', 'incr', midsouth)
secondary_77 = pf.Pickfile('pickdata_aquifer/77_wind_secondary.info', 'incr', midsouth)
secondary_78 = pf.Pickfile('pickdata_aquifer/78_wind_secondary.info', 'incr', midspread)
secondary_79 = pf.Pickfile('pickdata_aquifer/79_wind_secondary.info', 'incr', midspread)
secondary_80 = pf.Pickfile('pickdata_aquifer/80_wind_secondary.info', 'incr', 60, 3)
secondary_81 = pf.Pickfile('pickdata_aquifer/81_wind_secondary.info', 'incr', 60, 3)
secondary_82 = pf.Pickfile('pickdata_aquifer/82_wind_secondary.info', 'incr', 60, 3)
secondary_lithic_fullstack = pf.Pickfile('pickdata_lithic/fullstack_stacked_secondary.info', 'incr', 0, 3, maxrows=24)

# badtr = 6
# primary_29 = pf.Pickfile('pickdata/29_primary.info', 23)
# primary_29.fliptrace(badtr)
# secondary_29 = pf.Pickfile('pickdata/29_secondary.info', 23)
# secondary_29.fliptrace(badtr)
# primary_30 = pf.Pickfile('pickdata/30_primary.info', 23)
# primary_30.fliptrace(badtr)
# secondary_30 = pf.Pickfile('pickdata/30_secondary.info', 23)
# secondary_30.fliptrace(badtr)
# primary_31 = pf.Pickfile('pickdata/31_primary.info', 23)
# primary_31.fliptrace(badtr)
# secondary_31 = pf.Pickfile('pickdata/31_secondary.info', 23)
# secondary_31.fliptrace(badtr)
#
# primary_32 = pf.Pickfile('pickdata/32_primary.info', 20)
# primary_32.fliptrace(badtr)
# secondary_32 = pf.Pickfile('pickdata/32_secondary.info', 20)
# secondary_32.fliptrace(badtr)
# primary_33 = pf.Pickfile('pickdata/33_primary.info', 20)
# primary_33.fliptrace(badtr)
# secondary_33 = pf.Pickfile('pickdata/33_secondary.info', 20)
# secondary_33.fliptrace(badtr)
# primary_34 = pf.Pickfile('pickdata/34_primary.info', 20)
# primary_34.fliptrace(badtr)
# primary_fullstack = pf.Pickfile('pickdata/fullstack_stacked_primary.info', 0)
# secondary_fullstack = pf.Pickfile('pickdata/fullstack_stacked_secondary.info', 0)
#
# # We'll create an array that lists each shot number and the shot location corresponding to it
# shot_loc = np.array([
#     [29, 23],
#     [30, 23],
#     [31, 23],
#     [32, 20],
#     [33, 20],
#     # [34, 20],
#     [35, 17],
#     [36, 17],
#     [37, 14],
#     [39, 14],
#     # [42, 11], delayed
#     [43, 7],
#     [44, 7],
#     [45, 4],
#     # [46, 4], delayed
#     [47, 4],
#     [50, 0]
# ])
#
# shot_loc = np.array([
#     [72, 195],
#     [73, 195],
#     [74, 195],
#     [76, midsouth],
#     [77, midsouth],
#     [78, midspread],
#     [79, midspread],
#     [80, 60],
#     [81, 60],
#     [82, 60]
#   ])

# # Next we'll load in the shot locations from their files based on our shot_loc
# # The files are provided in the format #_primary.info and #_secondary.info
# shot_loc_data_primary = np.array([])
# for i in shot_loc:
#     pick = pf.Pickfile('pickdata_aquifer/' + str(i[0]) + '_wind_primary.info', 'incr', i[1])
#     # pick.fliptrace(badtr)
#     shot_loc_data_primary = np.append(shot_loc_data_primary, pick)
#
# shot_loc_data_secondary = np.array([])
# for i in shot_loc:
#     pick = pf.Pickfile('pickdata_aquifer/' + str(i[0]) + '_wind_secondary.info', 'incr', i[1])
#     # pick.fliptrace(badtr)
#     shot_loc_data_secondary = np.append(shot_loc_data_secondary, pick)


attn_opt, attn_cov = attn_fit([primary_lithic_fullstack])
# Next we'll fit attenuation using picks from the shot_loc_data_primary array
# attn_opt_new, attn_cov = attn_fit(shot_loc_data_primary)


# def primary_amp(primary, attn_coeff, inc_angle):
#     geom_corr = np.cos(inc_angle)/primary.dist_no_outliers
#     attn_corr = np.exp(-attn_coeff*primary.dist_no_outliers)
#     return primary.min_no_outliers/attn_corr/geom_corr*100
def primary_amp(primary, attn_coeff, inc_angle):
    geom_corr = np.cos(inc_angle)/primary.dist
    attn_corr = np.exp(-attn_coeff*primary.dist)
    return primary.min/attn_corr/geom_corr*100

inversion_results = inv1_fit([primary_lithic_fullstack])
# incidence_angle = inc_angle(primary_fullstack, inversion_results[0])
incidence_angle = inc_angle(primary_72, inversion_results[0])

# amp_stack = primary_amp(primary_stack, attn_opt[0], incidence_angle)
# amp_29 = primary_amp(primary_29, attn_opt[0], incidence_angle)
# amp_30 = primary_amp(primary_30, attn_opt[0], incidence_angle)
# amp_31 = primary_amp(primary_31, attn_opt[0], incidence_angle)
# amp_32 = primary_amp(primary_32, attn_opt[0], incidence_angle)
# amp_33 = primary_amp(primary_33, attn_opt[0], incidence_angle)
# plt.plot(primary_29.dist_no_outliers, amp_29)
# plt.plot(primary_30.dist_no_outliers, amp_30)
# plt.plot(primary_31.dist_no_outliers, amp_31)
# plt.show()


def reflectivity(primary, secondary, attn_coeff, polarity='max'):
    path_length = 2*np.sqrt(primary.dist**2 + 450**2)
    geom_corr = np.cos(primary.angle)/path_length
    attn_corr = np.exp(-attn_coeff*primary.dist)
    if polarity == 'max':
        return secondary.max/primary_amp(primary, attn_coeff, 0.0527)/attn_corr/geom_corr
    elif polarity == 'min':
        return secondary.min/primary_amp(primary, attn_coeff, 0.0527)/attn_corr/geom_corr


ref_29 = reflectivity(primary_29, secondary_29, attn_opt[0], polarity='min')
ref_30 = reflectivity(primary_30, secondary_30, attn_opt[0], polarity='min')
ref_31 = reflectivity(primary_31, secondary_31, attn_opt[0], polarity='min')
ref_lithic_stack = reflectivity(primary_lithic_fullstack, secondary_lithic_fullstack, attn_opt[0], polarity='min')
# ref_stack = reflectivity(primary_stack, secondary_stack, attn_opt[0])
ref_all = []
for i in range(len([primary_lithic_fullstack])):
    ref_temp = np.array(reflectivity([primary_lithic_fullstack][i], [secondary_lithic_fullstack][i], attn_opt[0]))
    ref_all.append(ref_temp)


# # Plot reflectivity as a function of angle
# for i in range(len(shot_loc_data_primary)):
#     # Make a scatter point with 2px radius
#     plt.scatter(np.rad2deg(shot_loc_data_primary[i].angle_no_outliers), ref_all[i], zorder=0, s=2)
# plt.plot(np.rad2deg(primary_32.angle_no_outliers), ref_32)
# plt.plot(np.rad2deg(primary_33.angle_no_outliers), ref_33)
# plt.scatter(np.rad2deg(primary_fullstack.angle_no_outliers), ref_stack, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_72.angle), ref_72, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_73.angle), ref_73, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_74.angle), ref_74, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_76.angle), ref_76, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_77.angle), ref_77, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_78.angle), ref_78, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_79.angle), ref_79, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_80.angle), ref_80, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_81.angle), ref_81, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_82.angle), ref_82, zorder=1, s=20)
plt.scatter(np.rad2deg(primary_29.angle), ref_29, zorder=1, s=20)
plt.scatter(np.rad2deg(primary_30.angle), ref_30, zorder=1, s=20)
plt.scatter(np.rad2deg(primary_31.angle), ref_31, zorder=1, s=20)
plt.scatter(np.rad2deg(primary_lithic_fullstack.angle), ref_lithic_stack, zorder=1, s=20)
# plt.scatter(np.rad2deg(primary_72.angle), ref_stack, zorder=1, s=20)
# # plt.legend(['29', '30', '31'])
plt.ylim([-0.5, 1])
plt.title('Reflectivity as fxn of angle')
plt.ylabel('Reflectivity')
plt.xlabel('Angle (deg)')
plt.grid()
plt.show()


depth = inv1_depth(np.arange(1, 200, 1), inversion_results[0])


def refl_time(offset, angle, velocity, depth=405):
    refl_timing = 2*np.sqrt(
                            (depth ** 2) +
                            ((offset/2)**2 * np.cos(np.deg2rad(angle))**2)
    ) / velocity
    return refl_timing

def zerodip_depth(reflectiontiming, offset, velocity):
    return np.sqrt((reflectiontiming*velocity/2)**2 - (offset/2)**2)
# Plot theoretical correct time difference between primary and secondary
# We'll do this using the primary arrival travel time curve generated by the inversion
refl_tdiff = np.array([])
refl_tdiff_inv = np.array([])
refl_timingarray_0deg = np.array([])
refl_timingarray_10deg = np.array([])
refl_timingarray_20deg = np.array([])
refl_deep = np.array([])
for i in range(1, 200, 1):
    # refl_timing = 2*np.sqrt(i**2 + 405**2)/3830
    refl_timingarray_0deg = np.append(refl_timingarray_0deg, refl_time(i, 0, 3600, 405))
    refl_timingarray_10deg = np.append(refl_timingarray_10deg, refl_time(i, -10, 3600, 405))
    refl_timingarray_20deg = np.append(refl_timingarray_20deg, refl_time(i, -20, 3600, 405))
    # refl_tdiff = np.append(refl_tdiff, refl_time(i, 0, 3600, 405) - i/3600)
    refl_deep = np.append(refl_deep, refl_time(i, 0, 3710, 477))
    # refl_tdiff_inv = np.append(refl_tdiff_inv, refl_timing - inv1(i, inversion_results[0][0], inversion_results[0][1],
    # inversion_results[0][2], inversion_results[0][3], inversion_results[0][4]))

# # Plot data vs theoretical reflection traveltime
# # plt.plot(primary_stack.dist, secondary_stack.tmin)
# #plt.plot(primary_stack.dist, secondary_stack.tmax)
# plt.plot(primary_72.dist, secondary_72.tmax)
# plt.plot(primary_73.dist, secondary_73.tmax)
# plt.plot(primary_74.dist, secondary_74.tmax)
# plt.plot(primary_76.dist, secondary_76.tmax)
# plt.plot(primary_77.dist, secondary_77.tmax)
# plt.plot(primary_78.dist, secondary_78.tmax)
# plt.plot(primary_79.dist, secondary_79.tmax)
# plt.plot(primary_80.dist, secondary_80.tmax)
# plt.plot(primary_81.dist, secondary_81.tmax)
# plt.plot(primary_82.dist, secondary_82.tmax)
# plt.plot(np.arange(1,200,1), refl_timingarray_0deg, zorder=0)
# plt.plot(np.arange(1,200,1), refl_timingarray_10deg, zorder=0)
# plt.plot(np.arange(1,200,1), refl_timingarray_20deg, zorder=0)
# # plt.legend(['Data', 'Theoretical'])
# plt.title('Data vs theoretical reflection traveltime')
# # plt.legend(['Shot 1', 'Shot 2', 'Shot 3', '0 deg bed', '10 deg bed', '20 deg bed'])
# plt.xlabel('Offset (m)')
# plt.ylabel('Traveltime (s)')
# plt.show()

# # Plot zero dip reflector depth
# plt.title('Reflector depth assuming zero reflector dip')
# #plt.plot(primary_stack.dist/2, zerodip_depth(secondary_stack.tmin, secondary_stack.dist, 3800))
# plt.plot(primary_73.dist/2, zerodip_depth(secondary_73.tmax, secondary_73.dist, 3650))
# plt.plot(primary_74.dist/2, zerodip_depth(secondary_74.tmax, secondary_74.dist, 3650))
# plt.ylim(400, 450)
# plt.show()


# # Plot direct arrival time vs offset
# plt.plot(primary_72.dist, primary_72.tmin)
# plt.plot(primary_73.dist, primary_73.tmin)
# plt.plot(primary_74.dist, primary_74.tmin)
# plt.plot(primary_76.dist, primary_76.tmin)
# plt.plot(primary_77.dist, primary_77.tmin)
# plt.plot(primary_78.dist, primary_78.tmin)
# plt.plot(primary_79.dist, primary_79.tmin)
# plt.plot(primary_80.dist, primary_80.tmin)
# plt.plot(primary_81.dist, primary_81.tmin)
# plt.plot(primary_82.dist, primary_82.tmin)
# plt.title('Direct arrival time vs offset')
# plt.xlabel('Offset (m)')
# plt.ylabel('Traveltime (s)')
# plt.show()

# Plot traveltime of lithic reflector vs offset
plt.plot(primary_lithic_fullstack.dist_no_outliers, secondary_lithic_fullstack.tmin_no_outliers)
plt.plot(primary_29.dist_no_outliers, secondary_29.tmin_no_outliers)
plt.plot(primary_30.dist_no_outliers, secondary_30.tmin_no_outliers)
plt.plot(primary_31.dist_no_outliers, secondary_31.tmin_no_outliers)
plt.plot(primary_33.dist_no_outliers, secondary_33.tmin_no_outliers)
plt.plot(primary_34.dist_no_outliers, secondary_34.tmin_no_outliers)
plt.xlabel('Offset (m)')
plt.ylabel('Traveltime (s)')
plt.plot(np.arange(1,200,1), refl_deep, zorder=0)
plt.show()

gs.depthvel_gridsearch_plot([primary_lithic_fullstack], [secondary_lithic_fullstack], prior=[3710, 477])