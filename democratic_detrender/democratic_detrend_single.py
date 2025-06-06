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
        "object", type=str, help='TESS or Kepler identifier. ex: "toi-2088". In this case, serves purely to label plots.'
    )

    parser.add_argument('planet_num', type=int, help='Number of planet. In this case, serves purely to label plots and subdirectories.')

    parser.add_argument("transit_num", type=int, help='Number of inputted transit. In this, serves purely to label plots.')

    parser.add_argument('path_to_lightcurve', help='Path that points to lightcurve of transit, including file name.')

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
        help="Optionally input period. Defaults to None. None option is useful if period is wrong."
    )

    parser.add_argument(
        "-t",
        "--t0",
        default=None,
        help='Optionally input t0 (mid-transit time). Otherwise None but will break. Should be in units of' \
            'TESS BJD - 2457000.'
    )

    parser.add_argument(
        "-du",
        "--duration",
        default=None,
        help="Optionally input duration.",
    )

    parser.add_argument(
        "-mw", "--mask_width", default=1.3, help="Sets mask width. Default is 1.3."
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

def get_single_user_lightcurve(lightcurve_path, save_to_dir, planet_num, transit_num, already_normalized, depth, period,
                           t0, duration, mask_width, show_plots):
       
    # read in light curve as pandas dataframe
    # assuming three columns: xs (time), ys (flux), and ys_err (flux error)

    print(f"Now loading light curve from: {lightcurve_path}")

    lc_df = pd.read_csv(lightcurve_path)

    # assumes lightcurve file has three columns in the following order: time (x) array, flux (y) array, flux error (yerr) array
    xs = lc_df.iloc[:, 0].values
    ys = lc_df.iloc[:, 1].values
    ys_err = lc_df.iloc[:, 2].values

    if not already_normalized:
        # normalize
        mu = np.median(ys)
        ys = ys / mu - 1
        ys_err = ys_err / mu
        
    return xs, ys, ys_err

def prepare_single_transit_data(xs, ys, ys_err, t0, period, duration, mask_width=1.3):

    # first convert to numpy arrays
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ys_err = np.asarray(ys_err, dtype=float)

    # then determine cadence
    cadence = determine_cadence(xs)

    # create masks
    duration_in_days = duration / 24.0 # convert to days
    half_width = duration_in_days * mask_width / 2.0
    mask = np.abs(xs - t0) <= half_width
    mask_fitted_planet = mask.copy()

    # package as a single epoch as the original functions expect lists

    x_epochs = [xs]
    y_epochs = [ys]
    yerr_epochs = [ys_err]
    mask_epochs = [mask]
    mask_fitted_planet_epochs = [mask_fitted_planet]

    # create t0s array
    if t0 is not None:
        t0s = np.asarray([t0])
    else:
        t0s = np.asarray([np.median(xs)])

    # and then period (use a v large value if not provided)
    if period is None:
        period = 1000.0 # days

    # duration now

    if duration is None:
        if np.any(mask):
            # Estimate from mask
            transit_indices = np.where(mask)[0]
            duration = (xs[transit_indices[-1]] - xs[transit_indices[0]]) * 24
        else:
            duration = 3.0  # default 3 hours

    # then find local window bounds
    if t0 is not None:
        transit_idx = np.argmin(np.abs(xs - t0))
        local_start = max(0, transit_idx - int(3 * duration / (24 * cadence)))
        local_end = min(len(xs)-1, transit_idx + int(3 * duration / (24 * cadence)))
        local_start_x = xs[local_start]
        local_end_x = xs[local_end]
    else:
        local_start_x = xs[0]
        local_end_x = xs[-1]
    
    # now create local window for local detrending
    local_window_size = float(6 * duration / 24.0) / period
    # local_data corresponds to
    ''' 
        local_x_epochs,
        local_y_epochs,
        local_yerr_epochs,
        local_mask_epochs,
        local_mask_fitted_planet_epochs'''
    local_data = split_around_transits(
        xs, ys, ys_err, mask, mask_fitted_planet,
        t0s, local_window_size, period
    )

    return [
        xs, ys, ys_err,
        mask, mask_fitted_planet,
        x_epochs, y_epochs, yerr_epochs,
        mask_epochs, mask_fitted_planet_epochs,
        t0s, period, duration, cadence,
        local_start_x, local_end_x, local_data
    ]
    

def single_local(xs, ys, ys_err, t0, period, duration):

    # first prepare data to match what local method is expecting
    epochdata = prepare_single_transit_data(xs, ys, ys_err, t0, period, duration)

    # then feed into local method function
    local_detrended = local_method(
        epochdata[5], epochdata[6], epochdata[7], # x_epochs, y_epochs, yerr_epochs
        epochdata[8], epochdata[9],          # mask_epochs, mask_fitted_planet_epochs
        epochdata[10], epochdata[12], epochdata[11]  # t0s, duration, period
    )

    return local_detrended, epochdata

def single_polyam(xs, ys, ys_err, t0, period, duration):

    # first prepare data to match what polyAM method is expecting
    epochdata = prepare_single_transit_data(xs,ys, ys_err, t0, period, duration)

    # interpolate model to all points
    try:

        # call polyAM_iterative
        poly = polyAM_iterative(
            epochdata[0], epochdata[1], epochdata[3], epochdata[4], # xs, ys, ys_err, mask, mask_fitted_planet
            epochdata[14], epochdata[15] # local_start_x, local_end _x
        )

        poly_interp = interp1d(
            epochdata[0][~epochdata[3]], poly[0], bounds_error=False, fill_value='extrapolate'
        )
        best_model = poly_interp(epochdata[0])
        DW = poly[2]

        # detrend 
        detrended = get_detrended_lc(epochdata[1], best_model)

    except:
        print(f'polyAM failed for this epoch')
        detrended = np.asarray([])
        DW = None

    # then add linear detrending
    try:
        linear_model = polyAM_function(epochdata[0][~epochdata[3]], detrended[~epochdata[3]], 1)
        linear_interp = interp1d(
            epochdata[0][~epochdata[3]], linear_model,
            bounds_error = False, fill_value = 'extrapolate'
        )
        final_linear = linear_interp(epochdata[0])
        final_detrended = get_detrended_lc(detrended, final_linear)
    except:
        print("polyAM failed for this epoch")
        final_detrended = np.asarray([])

    return final_detrended, DW, epochdata


def single_gp(xs, ys, ys_err, t0, period, duration):

    epochdata = prepare_single_transit_data(xs, ys, ys_err, t0, period, duration)

    # use original gp method
    detrended = gp_method(
        epochdata[5], epochdata[6], epochdata[7], # x_epochs, y_epochs, yerr_epochs
        epochdata[8], epochdata[9],          # mask_epochs, mask_fitted_planet_epochs
        epochdata[10], epochdata[12], epochdata[11]  # t0s, duration, period
    )

    return detrended, epochdata

def single_cofiam(xs, ys, ys_err, t0, period, duration):

    epochdata = prepare_single_transit_data(xs, ys, ys_err, t0, period, duration)

    try:
        # Call original cofiam_iterative
        cofiam = cofiam_iterative(
            epochdata[0], epochdata[1], epochdata[3], epochdata[4], # xs, ys, ys_err, mask, mask_fitted_planet
            epochdata[14], epochdata[15] # local_start_x, local_end _x
        )

        # interpolate model
        cofiam_interp = interp1d(
            epochdata[0][~epochdata[3]], cofiam[0], bounds_error=False, fill_value='extrapolate'
        )
        best_model = cofiam_interp(epochdata[0])
        DW = cofiam[2]
    except:
        print('CoFiAM failed for this epoch')
        detrended = np.asarray([])
        DW = None
    
    # detrend
    detrended = get_detrended_lc(epochdata[1], best_model)

    # add linear detrending
    try:
        linear_model = polyAM_function(epochdata[0][~epochdata[3]], detrended[~epochdata[3]], 1)
        linear_interp = interp1d(
            epochdata[0][~epochdata[3]], linear_model,
            bounds_error = False, fill_value = 'extrapolate'
        )
        final_linear = linear_interp(epochdata[0])
        final_detrended = get_detrended_lc(detrended, final_linear)
    except:
        print("CoFiAM failed for this epoch")
        final_detrended = np.asarray([])

    return final_detrended, DW, epochdata




def detrend_single_variable_methods(xs, ys, ys_err, t0, period, duration, save_to_directory, show_plots, detrend_methods):


    all_detrended = []
    all_results = []
    detrend_methods_success = []

  
    # below function returns:
    # [    xs, ys, ys_err,
    #     mask, mask_fitted_planet,
    #     x_epochs, y_epochs, yerr_epochs,
    #     mask_epochs, mask_fitted_planet_epochs,
    #     t0s, period, duration, cadence,
    #     local_start_x, local_end_x, local_data ]
    prepared_metadata = prepare_single_transit_data(xs, ys, ys_err, t0, period, duration)

    # local detrending
    if "local" in detrend_methods:
        start = time.time()
        # try
        print("")
        print("detrending via the local method")
        local_detrended, local_metadata = single_local(
            xs, ys, ys_err, t0, period, duration
        )

        # remove outliers in unmasksed local detrended lc
        local_x = np.concatenate(local_metadata[-1][0], axis=0, dtype=object)
        local_yerr = np.concatenate(local_metadata[-1][2], axis=0, dtype=object)
        cadence = local_metadata[13]
        t0s = local_metadata[10]
        period_processed = local_metadata[11]
        duration_processed = local_metadata[12]
        
        local_x_no_outliers, local_detrended_no_outliers = reject_outliers_everywhere(
            local_x, local_detrended, local_yerr, 5 * cadence, 5, 10
        )

        plot_individual_outliers(
            local_x,
            local_detrended,
            local_x_no_outliers,
            local_detrended_no_outliers,
            t0s, period,
            float(6*duration / (24.0) / period),
            0.009,
            save_to_directory + 'local_outliers.pdf'
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
        poly_detrended, poly_DWs, poly_metadata = single_polyam(
            xs, ys, ys_err, t0, period, duration
        )

        local_x = np.concatenate(poly_metadata[-1][0], axis=0, dtype=object)
        local_yerr = np.concatenate(poly_metadata[-1][2], axis=0, dtype=object)
        cadence = poly_metadata[13]
        t0s = poly_metadata[10]
        period_processed = poly_metadata[11]
        duration_processed = poly_metadata[12]

        # remove outliers in unmasked poly detrended lc
        poly_x_no_outliers, poly_detrended_no_outliers = reject_outliers_everywhere(
            local_x, poly_detrended, local_yerr, 5 * cadence, 5, 10
        )

        plot_individual_outliers(
            local_x,
            poly_detrended,
            poly_x_no_outliers,
            poly_detrended_no_outliers,
            t0s, period,
            float(6*duration / (24.0) / period),
            0.009,
            save_to_directory + 'polyAM_outliers.pdf'
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
        gp_detrended, gp_metadata = single_gp(
            xs, ys, ys_err, t0, period, duration
        )

        local_x = np.concatenate(gp_metadata[-1][0], axis=0, dtype=object)
        local_yerr = np.concatenate(gp_metadata[-1][2], axis=0, dtype=object)
        cadence = gp_metadata[13]
        t0s = gp_metadata[10]
        period_processed = gp_metadata[11]
        duration_processed = gp_metadata[12]

        # remove outliers in unmasked poly detrended lc
        gp_x_no_outliers, gp_detrended_no_outliers = reject_outliers_everywhere(
            local_x, gp_detrended, local_yerr, 5 * cadence, 5, 10
        )

        plot_individual_outliers(
            local_x,
            gp_detrended,
            gp_x_no_outliers,
            gp_detrended_no_outliers,
            t0s, period,
            float(6*duration / (24.0) / period),
            0.009,
            save_to_directory + 'GP_outliers.pdf'
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
        cofiam_detrended, cofiam_DWs, cofiam_metadata = single_cofiam(
            xs, ys, ys_err, t0, period, duration
        )

        local_x = np.concatenate(cofiam_metadata[-1][0], axis=0, dtype=object)
        local_yerr = np.concatenate(cofiam_metadata[-1][2], axis=0, dtype=object)
        local_y = np.concatenate(cofiam_metadata[-1][1], axis=0, dtype=object)
        local_mask = np.concatenate(cofiam_metadata[-1][3], axis=0, dtype=object)
        local_mask_fitted_planet = np.concatenate(cofiam_metadata[-1][4], axis=0, dtype=object)
        cadence = cofiam_metadata[13]
        t0s = cofiam_metadata[10]
        period_processed = cofiam_metadata[11]
        duration_processed = cofiam_metadata[12]

        # remove outliers in unmasked poly detrended lc
        cofiam_x_no_outliers, cofiam_detrended_no_outliers = reject_outliers_everywhere(
            local_x, cofiam_detrended, local_yerr, 5 * cadence, 5, 10
        )

        plot_individual_outliers(
            local_x,
            cofiam_detrended,
            cofiam_x_no_outliers,
            cofiam_detrended_no_outliers,
            t0s, period,
            float(6*duration / (24.0) / period),
            0.009,
            save_to_directory + 'CoFiAM_outliers.pdf'
        )
        
        end = time.time()
        print(
            "CoFiAM detrending completed in "
            + str(np.round(end - start, 2))
            + " seconds"
        )


    local_x = np.concatenate(prepared_metadata[-1][0], axis=0, dtype=object)
    local_yerr = np.concatenate(prepared_metadata[-1][2], axis=0, dtype=object)
    local_y = np.concatenate(prepared_metadata[-1][1], axis=0, dtype=object)
    local_mask = np.concatenate(prepared_metadata[-1][3], axis=0, dtype=object)
    local_mask_fitted_planet = np.concatenate(prepared_metadata[-1][4], axis=0, dtype=object)

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
    input_id = args.object
    input_planet_number = args.planet_num
    input_transit_number = args.transit_num
    input_lightcurve_dir = args.path_to_lightcurve
    already_normalized = args.already_normalized
    input_save_to_dir = args.save_to_dir
    input_depth = float(args.depth)
    input_period = float(args.period) if args.period else None
    input_t0 = float(args.t0) if args.t0 else None
    input_duration = float(args.duration) if args.duration else None
    input_mask_width = float(args.mask_width)
    input_show_plots = ast.literal_eval(args.show_plots)
    input_polyAM = args.polyAM
    input_CoFiAM = args.CoFiAM
    input_local = args.local
    input_GP = args.GP

    if input_t0 is None:
        print("Error: t0 (mid-transit time) is required")
        sys.exit(1)
        
    
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

    # read in single light curve
    time, flux, flux_err = get_single_user_lightcurve(
        input_lightcurve_dir, input_save_to_dir, input_planet_number, input_transit_number, already_normalized, \
            input_depth, input_period, input_t0, input_duration, input_mask_width, input_show_plots
    )

           
    if not input_save_to_dir:
        save_to_dir = os.path.dirname(input_lightcurve_dir) # save stuff in same directory as lightcurve file
    else:
        save_to_dir = input_save_to_dir
    print(f'Will save detrended data to {save_to_dir}.')

    # create detrended data folders
    path = os.path.join(save_to_dir, f'detrended_transit_{input_transit_number}')

    os.makedirs(path, exist_ok=True)

    # now to detrend

    detrend_methods, output = detrend_single_variable_methods(time, flux, flux_err, input_t0, input_period, input_duration, path+'/', input_show_plots, input_detrend_methods)

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
    mask_detrended = output[3]

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

    detrend_df.to_csv(path + "/" + "detrended.csv")


    # plot all detrended data
    plot_detrended_lc(
        x_detrended,
        y_detrended,
        detrend_label,
        [input_t0],
        float(6 * input_duration / (24.0)) / input_period,
        input_period,
        colors,
        input_duration,
        depth=input_depth,
        figname=path + "/" + "individual_detrended.pdf",
    )

    # plot method marginalized detrended data
    plot_detrended_lc(
        x_detrended,
        [method_marg_detrended],
        ["method marg"],
        [input_t0],
        float(6 * input_duration / (24.0)) / input_period,
        input_period,
        ["k"],
        input_duration,
        depth=input_depth,
        figname=path + "/" + "method_marg_detrended.pdf",
    )

    # plot binned phase folded lightcurve
    plot_phase_fold_lc(
        x_detrended,
        method_marg_detrended,
        input_period,
        input_t0,
        20,
        figname=path + "/" + "phase_folded.pdf",
    )

if __name__ == "__main__":
    single_main()