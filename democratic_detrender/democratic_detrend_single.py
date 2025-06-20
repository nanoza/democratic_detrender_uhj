# this .py file created by Diana Solano-Oropeza at Cornell University on June 3, 2025
# detrends one transit out of many, one of a time
# democratic_detrend_single.py

import time
from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exoplanet as xo
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation
from matplotlib.widgets import Slider, Button
import sys, argparse
import os
import warnings
import ast
# detrender functions
from find_flux_jumps import *
from get_lc import *
from helper_functions import *
from outlier_rejection import *
from manipulate_data import *
from plot import plot_detrended_lc, plot_phase_fold_lc 
from detrend import *

warnings.simplefilter("ignore", np.RankWarning)

def single_parse_arguments():
    '''single_parse_arguments 
    Takes in command line arguments

    Returns
    -------
    argparse.Namespace
        parsed arguments with attributes.
    '''

    parser = argparse.ArgumentParser(
        description="Given a user-inputted light curve for one of many transits of a particular object," \
        "detrends the light curve. Blind to flux type, mission, planet number, period, mid-transit time (t0)," \
        "transit duration, and problem times, and does not download any new data. Assumes one flux file with three columns:," \
        "time-array, flux-array, and error-flux-array.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "object_name", type=str, help='Exoplanetary object identifier. In this case, serves purely to label plots and subdirectories'
    )

    parser.add_argument('planet_num', type=int, help='Number of planet. In this case, serves purely to label plots and subdirectories.')

    parser.add_argument("transit_num", type=int, help='Number of inputted transit, assuming each input lightcurve is of one transit. In this, serves purely to label plots.')

    parser.add_argument('path_to_lightcurve', help='Path that points to lightcurve of transit, including file name.')

    parser.add_argument('--flux_type', default='DVT', help='Set flux type. Original supported PDC and SAP but not DVT. This version should support those in addition to DVT but '
                        'default will be DVT for now. Will incorporate PDC and SAP handling later.' )

    parser.add_argument('--already_normalized', action='store_true', help='If set, detrender assumes the flux is already normalized and thus' \
        'does not normalize it any further.')

    parser.add_argument(
        "--save_to_dir",
        type=str,
        default=None,
        help="Directory path to save csvs and figures to. By default, saves to same directory as the input lightcurve.",
    )

    parser.add_argument(
        "-d",
        "--depth",
        default=0.01,
        help="Sets depth of detrended plot. Default is 0.02.",
    )

    parser.add_argument(
        "--p",
        "--period",
        default=None,
        help="Optionally input period. Defaults to None. None option is useful if period is wrong. Will assume a period of 1000 days, need to check if this affects final analysis."
    )

    parser.add_argument(
        "-t",
        "--t0",
        default=None,
        help='Optionally input t0 (mid-transit time). Otherwise None but will assume input lightcurve is centered on transit and so will attempt to set the middle as the t0. Should be in units of' \
            'TESS BJD - 2457000. May need a guess t0 per transit to work properly.'
    )

    parser.add_argument(
        "-du",
        "--duration",
        default=None,
        help="Optionally input duration. Helps set transit mask. INPUT IN UNITS OF DAYS!!!",
    )

    parser.add_argument(
        "-mw", "--mask_width", default=1.3, help="Sets the number by which to multiply duration with to create the transit mask. Default is 1.3."
    )

    parser.add_argument(
        "-s",
        "--show_plots",
        default="True",
        help="Set whether to show non-problem-time plots.",
    )

    parser.add_argument(
        "--polyAM", default="True", help="detrend via polyAM...True or False"
    )
    parser.add_argument(
        "--CoFiAM", default="True", help="detrend via CoFiAM...True or False"
    )
    parser.add_argument("--local", default="True", help="detrend via local...True or False")
    parser.add_argument("--GP", default="True", help="detrend via GP...True or False")

    return parser.parse_args()

def prepare_ephemeris(user_period, user_t0, user_duration, xs, mask_width=1.3):

    # set t0 to median of time array if None
    if user_t0 is None:
        # t0 = np.median(xs)
        xs_midpoint = len(xs) // 2 # assuming input transit data is already centered on transit
        t0 = xs[xs_midpoint]
        print(f'No t0 found. Assuming input data is centered around transit, meaning t0 = {t0}.')
    else:
        t0 = user_t0

    # set to duration to either whole transit (might not detrend anything bc it thinks everything is the transit) or 2 hours
    if user_duration is None:
        # create temporary mask
        temp_duration = np.max(xs) - np.min(xs)
        temp_duration_in_days = temp_duration / 24.0 # convert to hours
        temp_half_width = temp_duration_in_days * mask_width / 2.0
        temp_mask = np.abs(xs - t0) <= temp_half_width
        if np.any(temp_mask):
            # Estimate from mask
            transit_indices = np.where(temp_mask)[0]
            duration = (xs[transit_indices[-1]] - xs[transit_indices[0]]) * 24
        else:
            duration = 2.0  # default 2 hours
    else:
        duration = user_duration

    # and then if period is None use a v large value if not provided
    if user_period is None:
        period = 1000.0 # days
    else:
        period = user_period

    return period, t0, duration


def process_single_user_lightcurve(lightcurve_path, save_to_directory, planet_num, transit_num, already_normalized, depth, period,
                           t0, duration, mask_width, show_plots, path, flux_type, objectname):
       
    # read in light curve as pandas dataframe
    # assuming three columns: xs (time), ys (flux), and ys_err (flux error)
    
    print(f"Now loading light curve from: {lightcurve_path}")

    lc_df = pd.read_csv(lightcurve_path)

    # assumes lightcurve file has three columns in the following order: time (x) array, flux (y) array, flux error (yerr) array
    xs = lc_df.iloc[:, 0].values
    ys = lc_df.iloc[:, 1].values
    ys_err = lc_df.iloc[:, 2].values

    # prepare ephermeris
    [period, t0, duration] = prepare_ephemeris(period, t0, duration, xs)

    t0s = [t0] # to make compatible with detrender functions

    # starting to diverge from democratic_detrender here a bit
    # may need to create masks later after first determining a neutral way to determine the transit without ephermeris
    # ok we doooo neeed transit duration though, or at least a guess
    # creating transit mask based off 1.3 * duration guess and a mid-transit time guess

    xs_array = np.asarray(xs) # time
    ys_array = np.asarray(ys) # lc
    ys_err_array = np.asarray(ys_err) # lc_err

    left_point_mask = t0 - mask_width*duration
    right_point_mask = t0 + mask_width*duration

    mask = (xs_array > left_point_mask) & (xs_array < right_point_mask) # np.array of booleans that flag whether data is in transit
    mask_fitted_planet = mask # working on one planet at a time, assuming one planet caught by transit data
    
    cadence = determine_cadence(xs)

    quarters = []
    quarters.append([np.min(xs), np.max(xs)]) # should be just the one sector (?)

    # end times of quarters
    quarters_end = [el[-1] for el in quarters]

    if not already_normalized:
        # normalize around 0
        mu = np.median(ys)
        ys = ys / mu - 1
        ys_err = ys_err / mu
    
    # reject outliers out of transit
    (
        time_out,
        flux_out,
        flux_err_out,
        mask_out,
        mask_fitted_planet_out,
        moving_median,
    ) = reject_outliers_out_of_transit(
        xs_array, ys_array, ys_err_array, np.asarray(mask,dtype=bool), np.asarray(mask_fitted_planet,dtype=bool), 30 * cadence, 3
    )

    plot_outliers(
        xs_array,
        ys_array,
        time_out,
        flux_out,
        moving_median,
        quarters_end,
        path + '/' +  str(flux_type) + "_" + "outliers.pdf",
        objectname
    )
    if show_plots:
        plt.show()

    (
        x_quarters,
        y_quarters,
        yerr_quarters,
        mask_quarters,
        mask_fitted_planet_quarters,
    ) = split_around_problems(
        time_out, flux_out, flux_err_out, mask_out, mask_fitted_planet_out, quarters_end
    ) # basically reuse split around problems to split around quarters

    plot_split_data(
        x_quarters,
        y_quarters,
        t0s,
        path + '/' + str(flux_type) +  "_" + "quarters_split.pdf",
        objectname,
    )
    if show_plots:
        plt.show()

    (
        x_quarters_w_transits,
        y_quarters_w_transits,
        yerr_quarters_w_transits,
        mask_quarters_w_transits,
        mask_fitted_planet_quarters_w_transits,
    ) = find_quarters_with_transits(
        x_quarters,
        y_quarters,
        yerr_quarters,
        mask_quarters,
        mask_fitted_planet_quarters,
        t0s,
    )
    x_quarters_w_transits = np.concatenate(x_quarters_w_transits, axis=0, dtype=object)
    y_quarters_w_transits = np.concatenate(y_quarters_w_transits, axis=0, dtype=object)
    yerr_quarters_w_transits = np.concatenate(
        yerr_quarters_w_transits, axis=0, dtype=object
    )
    mask_quarters_w_transits = np.concatenate(
        mask_quarters_w_transits, axis=0, dtype=object
    )
    mask_fitted_planet_quarters_w_transits = np.concatenate(
        mask_fitted_planet_quarters_w_transits, axis=0, dtype=object
    )

    mask_quarters_w_transits = np.array(mask_quarters_w_transits, dtype=bool)
    mask_fitted_planet_quarters_w_transits = np.array(
        mask_fitted_planet_quarters_w_transits, dtype=bool
    )

    (
        x_transits,
        y_transits,
        yerr_transits,
        mask_transits,
        mask_fitted_planet_transits,
    ) = split_around_transits(
        x_quarters_w_transits,
        y_quarters_w_transits,
        yerr_quarters_w_transits,
        mask_quarters_w_transits,
        mask_fitted_planet_quarters_w_transits,
        t0s,
        1.0 / 2.0,
        period,
    )

    if len(mask_transits) == 1:
        mask_transits = np.array(mask_transits, dtype=bool)
        mask_fitted_planet_transits = np.array(mask_fitted_planet_transits, dtype=bool)

    x_epochs = np.concatenate(x_transits, axis=0, dtype=object)
    y_epochs = np.concatenate(y_transits, axis=0, dtype=object)
    yerr_epochs = np.concatenate(yerr_transits, axis=0, dtype=object)
    mask_epochs = np.concatenate(mask_transits, axis=0, dtype=object)
    mask_fitted_planet_epochs = np.concatenate(
        mask_fitted_planet_transits, axis=0, dtype=object
    )

    # put in bit that handles problem times here, need to figure out how to adapt it later
    problem_times = []

    # adapt bit that checks for problem times later, put here

    return (
        x_epochs,
        y_epochs,
        yerr_epochs,
        mask_epochs,
        mask_fitted_planet_epochs,
        problem_times,
        t0s,
        period,
        duration,
        cadence,
    )   


def detrend_single_variable_methods(x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs, problem_times, t0s, period, duration, cadence, save_to_directory, show_plots, detrend_methods, path,
                                    flux_type, objectname):


    # insert thing that handles clipping around problem times here, testing with pre-clipped data for now
    # adaptation of trim_jump_times
    # just use x_epochs, y_epochs, yerr_epochs, mask_epochs, mask_fitted_planet_epochs in place
    # x_trimmed = x_epochs
    # y_trimmed = y_epochs
    # yerr_trimmed = yerr_epochs
    # mask_trimmed = mask_epochs
    # mask_fitted_planet_trimmed = mask_fitted_planet_epochs
    (
        x_trimmed,
        y_trimmed,
        yerr_trimmed,
        mask_trimmed,
        mask_fitted_planet_trimmed,
    ) = trim_jump_times(
        x_epochs,
        y_epochs,
        yerr_epochs,
        mask_epochs,
        mask_fitted_planet_epochs,
        t0s,
        period,
        problem_times,
    )

    #### polyam, gp, cofiam friendly mask arrays ####

    friendly_mask_trimmed = []
    for boolean in range(len(mask_trimmed)):
        friendly_boolean = mask_trimmed[boolean].astype(bool)
        friendly_mask_trimmed.append(friendly_boolean)

    friendly_mask_fitted_planet_trimmed = []
    for boolean in range(len(mask_fitted_planet_trimmed)):
        friendly_boolean = mask_fitted_planet_trimmed[boolean].astype(bool)
        friendly_mask_fitted_planet_trimmed.append(friendly_boolean)

    friendly_x_trimmed = []
    for time_array in range(len(x_trimmed)):
        friendly_time_array = x_trimmed[time_array].astype(float)
        friendly_x_trimmed.append(friendly_time_array)

    friendly_y_trimmed = []
    for flux_array in range(len(y_trimmed)):
        friendly_flux_array = y_trimmed[flux_array].astype(float)
        friendly_y_trimmed.append(friendly_flux_array)

    friendly_yerr_trimmed = []
    for flux_err_array in range(len(yerr_trimmed)):
        friendly_flux_err_array = yerr_trimmed[flux_err_array].astype(float)
        friendly_yerr_trimmed.append(friendly_flux_err_array)

    # determine local window values, zoom in around local window
    (
        local_x_epochs,
        local_y_epochs,
        local_yerr_epochs,
        local_mask_epochs,
        local_mask_fitted_planet_epochs,
    ) = split_around_transits(
        np.concatenate(x_trimmed, axis=0, dtype=object),
        np.concatenate(y_trimmed, axis=0, dtype=object),
        np.concatenate(yerr_trimmed, axis=0, dtype=object),
        np.concatenate(mask_trimmed, axis=0, dtype=object),
        np.concatenate(mask_fitted_planet_trimmed, axis=0, dtype=object),
        t0s,
        float(6 * duration) / period, # omit / 24.0 part because input duration should be in units of days already
        period,
    )

    local_x = np.concatenate(local_x_epochs, axis=0, dtype=object)
    local_y = np.concatenate(local_y_epochs, axis=0, dtype=object)
    local_yerr = np.concatenate(local_yerr_epochs, axis=0, dtype=object)
    local_mask = np.concatenate(local_mask_epochs, axis=0, dtype=object)
    local_mask_fitted_planet = np.concatenate(
        local_mask_fitted_planet_epochs, axis=0, dtype=object
    )

    local_x = np.asarray(friendly_x_trimmed[0], dtype=float)
    local_yerr = np.asarray(friendly_yerr_trimmed[0], dtype=float)

    # local detrending
    if "local" in detrend_methods:
        start = time.time()
        # try
        print("")
        print("detrending via the local method")
        local_detrended = local_method_single(
            friendly_x_trimmed,
            friendly_y_trimmed,
            friendly_yerr_trimmed,
            friendly_mask_trimmed,
            friendly_mask_fitted_planet_trimmed,
            t0s,
            duration,
            period,
        )
        
        local_x_no_outliers, local_detrended_no_outliers = reject_outliers_everywhere(
            local_x, local_detrended, local_yerr, 5 * cadence, 5, 3 # increasing number times cadence tends to increase number of outliers
                                                                      # increasing sigma tends to decrease number of outliers
        )


        plot_individual_outliers(
            local_x,
            local_detrended,
            local_x_no_outliers,
            local_detrended_no_outliers,
            t0s, period,
            float(6*duration / period),
            0.009,
            path + '/' + str(flux_type) + '_' + 'local_outliers.pdf'
        )

        end = time.time()

        print(
            "local detrending completed in "
            + str(np.round(end - start, 2))
            + " seconds"
        )
        
    # polyAM detrending
    if 'polyAM' in detrend_methods:
        start = time.time()
        print("")
        print('detrending via the polyAM method')


        poly_detrended, poly_DWs = polynomial_method_single(
            friendly_x_trimmed,
            friendly_y_trimmed,
            friendly_yerr_trimmed,
            friendly_mask_trimmed,
            friendly_mask_fitted_planet_trimmed,
            t0s,
            duration,
            period,
            local_x_epochs,
        )

        # remove outliers in unmasked poly detrended lc
        poly_x_no_outliers, poly_detrended_no_outliers = reject_outliers_everywhere(
            local_x, poly_detrended, local_yerr, 5 * cadence, 5, 3
        )

        plot_individual_outliers(
            local_x,
            poly_detrended,
            poly_x_no_outliers,
            poly_detrended_no_outliers,
            t0s, period,
            float(6*duration / period),
            0.009,
            path + '/' + str(flux_type) + '_' + 'polyAM_outliers.pdf'
        )
        
        end = time.time()
        print(
            "polyAM detrending completed in "
            + str(np.round(end - start, 2))
            + " seconds"
        )

    # now for GP detrending
    if 'GP' in detrend_methods:
        start = time.time()
        print("")
        print("detrending via the GP method")
        gp_detrended = gp_method_single(
            friendly_x_trimmed,
            friendly_y_trimmed,
            friendly_yerr_trimmed,
            friendly_mask_trimmed,
            friendly_mask_fitted_planet_trimmed,
            t0s,
            duration,
            period,
        )


        # remove outliers in unmasked poly detrended lc
        gp_x_no_outliers, gp_detrended_no_outliers = reject_outliers_everywhere(
            local_x, gp_detrended, local_yerr, 5 * cadence, 5, 3
        )

        plot_individual_outliers(
            local_x,
            gp_detrended,
            gp_x_no_outliers,
            gp_detrended_no_outliers,
            t0s, period,
            float(6*duration / period),
            0.009,
            save_to_directory + '/' + str(flux_type) + '_' +'GP_outliers.pdf'
        )
        
        end = time.time()
        print(
            "GP detrending completed in "
            + str(np.round(end - start, 2))
            + " seconds"
        )

    # no for cofiam
    if "CoFiAM" in detrend_methods:
        start = time.time()
        print("")
        print('detrending via the CoFiAM method')
        cofiam_detrended, cofiam_DWs = cofiam_method_single(
            friendly_x_trimmed,
            friendly_y_trimmed,
            friendly_yerr_trimmed,
            friendly_mask_trimmed,
            friendly_mask_fitted_planet_trimmed,
            t0s,
            duration,
            period,
            local_x_epochs,
        )
        
        # remove outliers in unmasked poly detrended lc
        cofiam_x_no_outliers, cofiam_detrended_no_outliers = reject_outliers_everywhere(
            local_x, cofiam_detrended, local_yerr, 5 * cadence, 5, 3
        )

        plot_individual_outliers(
            local_x,
            cofiam_detrended,
            cofiam_x_no_outliers,
            cofiam_detrended_no_outliers,
            t0s, period,
            float(6*duration / period),
            0.009,
            save_to_directory + '/' + str(flux_type) + '_' + 'CoFiAM_outliers.pdf'
        )
        
        end = time.time()
        print(
            "CoFiAM detrending completed in "
            + str(np.round(end - start, 2))
            + " seconds"
        )

    output = [local_x, local_y, local_yerr, local_mask, local_mask_fitted_planet]
    nan_array = np.empty(np.shape(local_x))
    nan_array[:] = np.nan
    detrend_methods_out = []

    if "local" in detrend_methods:
        detrend_methods_out.append("local")
        output.append(local_detrended)
        output.append(local_x_no_outliers)
        output.append(local_detrended_no_outliers)
    else:
        output.append(nan_array)
        output.append(nan_array)
        output.append(nan_array)

    if "polyAM" in detrend_methods:
        detrend_methods_out.append("polyAM")
        output.append(poly_detrended)
        output.append(poly_x_no_outliers)
        output.append(poly_detrended_no_outliers)
    else:
        output.append(nan_array)
        output.append(nan_array)
        output.append(nan_array)

    if "GP" in detrend_methods:
        detrend_methods_out.append("GP")
        output.append(gp_detrended)
        output.append(gp_x_no_outliers)
        output.append(gp_detrended_no_outliers)
    else:
        output.append(nan_array)
        output.append(nan_array)
        output.append(nan_array)

    if "CoFiAM" in detrend_methods:
        detrend_methods_out.append("CoFiAM")
        output.append(cofiam_detrended)
        output.append(cofiam_x_no_outliers)
        output.append(cofiam_detrended_no_outliers)
    else:
        output.append(nan_array)
        output.append(nan_array)
        output.append(nan_array)

    return detrend_methods_out, output






def single_main():

    args = single_parse_arguments() # grab arguments
    
    # Extract arguments
    input_id = args.object_name
    input_planet_number = args.planet_num
    input_transit_number = args.transit_num
    input_lightcurve_dir = args.path_to_lightcurve
    input_flux_type = args.flux_type
    already_normalized = args.already_normalized
    input_save_to_dir = args.save_to_dir
    input_depth = float(args.depth)
    input_period = float(args.p) if args.p else None
    input_t0 = float(args.t0) if args.t0 else None
    input_duration = float(args.duration) if args.duration else None
    input_mask_width = float(args.mask_width)
    input_show_plots = ast.literal_eval(args.show_plots)
    input_polyAM = args.polyAM
    input_CoFiAM = args.CoFiAM
    input_local = args.local
    input_GP = args.GP

    # create list of detrend methods
    input_detrend_methods = []
    if input_GP == "True":
        input_detrend_methods.append("GP")
    if input_CoFiAM == "True":
        input_detrend_methods.append("CoFiAM")
    if input_polyAM == "True":
        input_detrend_methods.append("polyAM")
    if input_local == "True":
        input_detrend_methods.append("local")

    if not input_save_to_dir:
        save_to_dir = os.path.dirname(input_lightcurve_dir) # save stuff in same directory as lightcurve file
    else:
        save_to_dir = input_save_to_dir

    # create detrended data folders
    path = os.path.join(save_to_dir, f'detrended_transit_{input_transit_number}')

    print(f'Will save detrended data to {path}.')

    os.makedirs(path, exist_ok=True)

    # read in single light curve
    [
            x_epochs,
            y_epochs,
            yerr_epochs,
            mask_epochs,
            mask_fitted_planet_epochs,
            problem_times,
            t0s,
            period,
            duration,
            cadence,
        ] = process_single_user_lightcurve(
        input_lightcurve_dir, save_to_dir, input_planet_number, input_transit_number, already_normalized, \
            input_depth, input_period, input_t0, input_duration, input_mask_width, input_show_plots, path,
            input_flux_type, input_id
    )
    # now to detrend

    detrend_methods, output = detrend_single_variable_methods(x_epochs, y_epochs, yerr_epochs,
                                                              mask_epochs, mask_fitted_planet_epochs, problem_times,
                                                               t0s, period, duration, cadence, path+'/', input_show_plots, input_detrend_methods, path, input_flux_type, input_id)

    # now let's plot and save data

    green2, green1 = "#355E3B", "#18A558"
    blue2, blue1 = "#000080", "#4682B4"
    purple2, purple1 = "#2E0854", "#9370DB"
    red2, red1 = "#770737", "#EC8B80"

    colors = [red1, red2, blue1, blue2, green1, green2, purple1, purple2]

    y_detrended = [
        output[5], # local
        output[8], # polyam
        output[11], # gp
        output[14]  # cofiam
    ]

    detrend_label = [
        "local",
        "polyAM",
        "GP",
        "CoFiAM",
    ]

    y_detrended = np.array(y_detrended)
    y_detrended_transpose = y_detrended.T

    x_detrended = output[0]
    yerr_detrended = output[2] # local_yerr
    mask_detrended = mask_epochs

    method_marg_detrended = np.nanmedian(y_detrended_transpose, axis=1)
    MAD = median_abs_deviation(
        y_detrended_transpose, axis =1, scale = 1/1.486, nan_policy='omit'
    )
    yerr_detrended = np.sqrt(yerr_detrended.astype(float)**2 + MAD**2)

    # save detrend data as csv

    detrend_dict = {}

    detrend_dict["time"] = x_detrended
    detrend_dict["yerr"] = yerr_detrended
    detrend_dict["mask"] = mask_detrended
    detrend_dict["method marginalized"] = method_marg_detrended

    for ii in range(0, len(y_detrended)):
        detrend = y_detrended[ii]
        label = detrend_label[ii]
        detrend_dict[label] = detrend

    detrend_df = pd.DataFrame(detrend_dict)

    detrend_df.to_csv(path + "/" + str(input_flux_type) + '_' + "detrended.csv")

    # plot all detrended data
    plot_detrended_lc_single(
        x_detrended,
        y_detrended,
        detrend_label,
        t0s,
        float(6 * duration) / period,
        period,
        colors,
        duration,
        depth=input_depth,
        figname=path + "/" + str(input_flux_type) + '_' + "individual_detrended.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc_single(
        x_detrended,
        [method_marg_detrended],
        ["method marg"],
        t0s,
        float(6 * duration) / period,
        period,
        ["k"],
        duration,
        depth=input_depth,
        figname=path + "/" + str(input_flux_type) + '_' + "method_marg_detrended.pdf",
    )

    # now let's handle detrended data that has had outliers removed

    y_out_detrended = [
        output[7], # local
        output[10], # polyam
        output[13], # gp
        output[16]  # cofiam
    ]

    y_out_detrended = np.array(y_out_detrended)
    
    x_out_detrended = [
        output[6], # local
        output[9], # polyam
        output[12], # gp
        output[15]  # cofiam
    ]

    x_out_detrended = np.array(x_out_detrended)

    y_out_full_timegrid_detrended = []

    # for each detrended method
    for i in range(len(y_out_detrended)):

        y_out_detrended_i = y_out_detrended[i]
        x_out_detrended_i = x_out_detrended[i]
        yerr_detrended = output[2] # local_yerr
        mask_detrended = mask_epochs


        # add any missing y at x values back in as nans if theyve been removed as outliers
        y_out_detrended_i = add_nans_for_detrended_removed_outlier_data(x_detrended, x_out_detrended_i, y_out_detrended_i)
        y_out_full_timegrid_detrended.append(y_out_detrended_i)



    # then take the method marginalization
    y_out_full_timegrid_detrended_transpose = np.array(y_out_full_timegrid_detrended).T
    method_marg_out_detrended = np.nanmedian(y_out_full_timegrid_detrended_transpose, axis=1)
    MAD_out = median_abs_deviation(
        y_out_full_timegrid_detrended_transpose, axis =1, scale = 1/1.486, nan_policy='omit'
    )
    yerr_out_detrended = np.sqrt(yerr_detrended.astype(float)**2 + MAD_out**2)

    # save detrended no outlier data as txt file, comma-separated

    detrend_out_dict = {}

    detrend_out_dict["time"] = x_detrended
    # detrend_out_dict["mask"] = mask_detrended
    detrend_out_dict["method marginalized out"] = method_marg_out_detrended
    detrend_out_dict["yerr"] = yerr_out_detrended


    # for ii in range(0, len(y_out_full_timegrid_detrended)):
    #     detrend = y_out_full_timegrid_detrended[ii]
    #     label = detrend_label[ii]
    #     detrend_out_dict[label] = detrend

    detrend_out_df = pd.DataFrame(detrend_out_dict)
    detrend_out_df.to_csv(path + "/" + str(input_flux_type) + '_' + "detrended_no_outliers.txt", index=False)

    # plot all detrended without outliers data
    plot_detrended_lc_single(
        x_detrended, # use original x detrended as it contains all x values
        y_out_full_timegrid_detrended,
        detrend_label,
        t0s,
        float(6 * duration) / period,
        period,
        colors,
        duration,
        depth=input_depth,
        figname=path + "/" + str(input_flux_type) + '_' + "individual_detrended_no_outliers.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc_single(
        x_detrended, # use original x detrended as it contains all x values
        [method_marg_out_detrended],
        ["method marg"],
        t0s,
        float(6 * duration) / period,
        period,
        ["k"],
        duration,
        depth=input_depth,
        figname=path + "/" + str(input_flux_type) + '_' + "method_marg_detrended_no_outliers.pdf",
    )













if __name__ == "__main__":
    single_main()