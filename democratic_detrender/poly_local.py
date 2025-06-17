### Special thanks to Alex Teachey --> adapted from MoonPy package
### GitHub: https://github.com/alexteachey/MoonPy

import numpy as np
from manipulate_data import *
from helper_functions import *
from poly_AM import *
from plot import *


def BIC(model, data, errors, nparams):
    chi2 = np.nansum(((model - data) / errors) ** 2)
    BICval = nparams * np.log(len(data)) + chi2
    return BICval


### this function spits out the best fit line!
def polyLOC_function(times, fluxes, degree):

    times = np.array(times, dtype=float)
    fluxes = np.array(fluxes, dtype=float)

    poly_coeffs = np.polyfit(times, fluxes, degree)
    model = np.polyval(poly_coeffs, times)
    return model


def polyLOC_iterative(times, fluxes, errors, mask, max_degree=30, min_degree=1):
    ### this function utilizes polyLOC_function above, iterates it up to max_degree.
    ### max degree may be calculated using max_order function

    vals_to_min = []
    degs_to_try = np.arange(min_degree, max_degree + 1, 1)
    BICstats = []

    mask = np.array(mask, dtype=bool)
    for deg in degs_to_try:
        output_function = polyLOC_function(
            times[~mask], fluxes[~mask], deg
        )  ### this is the model
        residuals = fluxes[~mask] - output_function
        BICstat = BIC(output_function, fluxes[~mask], errors[~mask], deg + 1)
        BICstats.append(BICstat)

    BICstats = np.array(BICstats)

    best_degree = degs_to_try[np.argmin(BICstats)]
    best_BIC = BICstats[np.argmin(np.array(BICstats))]

    ### re-generate the function with the best degree

    best_model = polyLOC_function(times[~mask], fluxes[~mask], best_degree)

    return best_model, best_degree, best_BIC, max_degree


def local_method(
    x_epochs,
    y_epochs,
    yerr_epochs,
    mask_epochs,
    mask_fitted_planet_epochs,
    t0s,
    duration,
    period,
):

    from scipy.stats import median_absolute_deviation

    x = np.concatenate(x_epochs, axis=0)
    y = np.concatenate(y_epochs, axis=0)
    yerr = np.concatenate(yerr_epochs, axis=0)
    mask = np.concatenate(mask_epochs, axis=0, dtype=bool)
    mask_fitted_planet = np.concatenate(mask_fitted_planet_epochs, axis=0, dtype=bool)

    (
        x_local,
        y_local,
        yerr_local,
        mask_local,
        mask_fitted_planet_local,
    ) = split_around_transits(
        x,
        y,
        yerr,
        mask,
        mask_fitted_planet,
        t0s,
        float(6 * duration / (24.0)) / period,
        period,
    )

    local_mod = []

    x_all = []
    y_all = []
    yerr_all = []
    mask_all = []
    mask_fitted_planet_all = []

    for ii in range(0, len(x_local)):
        x_ii = np.array(x_local[ii], dtype=float)
        y_ii = np.array(y_local[ii], dtype=float)
        yerr_ii = np.array(yerr_local[ii], dtype=float)
        mask_ii = np.array(mask_local[ii], dtype=bool)
        mask_fitted_planet_all_ii = np.array(mask_fitted_planet_local[ii], dtype=bool)

        try:
            local = polyLOC_iterative(x_ii, y_ii, yerr_ii, mask_ii)

            polyLOC_interp = interp1d(
                x_ii[~mask_ii], local[0], bounds_error=False, fill_value="extrapolate"
            )
            best_model = polyLOC_interp(x_ii)

            local_mod.append(best_model)

        except:
            print("local failed for the " + str(ii) + "th epoch")
            # local failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(x_ii))
            nan_array[:] = np.nan

            local_mod.append(nan_array)

        x_all.extend(x_ii)
        y_all.extend(y_ii)
        yerr_all.extend(yerr_ii)
        mask_all.extend(mask_ii)
        mask_fitted_planet_all.extend(mask_fitted_planet_all_ii)

    # add a linear polynomial fit at the end
    model_linear = []
    y_out_detrended = []
    for ii in range(0, len(local_mod)):
        x_ii = np.array(x_local[ii], dtype=float)
        y_ii = np.array(y_local[ii], dtype=float)
        mask_ii = np.array(mask_local[ii], dtype=bool)
        model_ii = np.array(local_mod[ii], dtype=float)

        y_ii_detrended = get_detrended_lc(y_ii, model_ii)

        try:
            linear_ii = polyAM_function(x_ii[~mask_ii], y_ii_detrended[~mask_ii], 1)
            poly_interp = interp1d(
                x_ii[~mask_ii], linear_ii, bounds_error=False, fill_value="extrapolate"
            )
            model_ii_linear = poly_interp(x_ii)

            model_linear.append(model_ii_linear)

            y_ii_linear_detrended = get_detrended_lc(y_ii_detrended, model_ii_linear)
            y_out_detrended.append(y_ii_linear_detrended)

        except:
            print("local failed for the " + str(ii) + "th epoch")
            # local failed for this epoch, just add nans of the same size
            nan_array = np.empty(np.shape(x_ii))
            nan_array[:] = np.nan

            y_out_detrended.append(nan_array)

    detrended_lc = np.concatenate(y_out_detrended, axis=0)
    # detrended_x = np.concatenate(x_local, axis=0)

    return detrended_lc

def local_method_single(
    x_epochs,
    y_epochs,
    yerr_epochs,
    mask_epochs,
    mask_fitted_planet_epochs,
    t0s,
    duration,
    period,
):
    

    from scipy.stats import median_absolute_deviation

    x = x_epochs[0]
    y = y_epochs[0]
    yerr = yerr_epochs[0]
    mask = mask_epochs[0]
    mask_fitted_planet = mask_fitted_planet_epochs[0]

    # then convert to proper np arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    yerr = np.array(yerr, dtype=float)
    mask = np.array(mask, dtype=bool)
    mask_fitted_planet = np.array(mask_fitted_planet, dtype=bool)

    t0 = t0s[0]

    # define local window
    # duration_days = duration / 24.0
    window_half_width = 3 * duration  # 3 durations on each side = 6 total, assuming given duration is in days
    print('local')

    
    # Extract local window around transit
    local_mask = np.abs(x - t0) <= window_half_width
    print(len(local_mask))
    x_local = x[local_mask]
    y_local = y[local_mask]
    yerr_local = yerr[local_mask]
    mask_local = mask[local_mask]
    

    try:
        local = polyLOC_iterative(x_local, y_local, yerr_local, mask_local)
        # print(local)

        polyLOC_interp = interp1d(
            x_local[~mask_local], local[0], bounds_error=False, fill_value="extrapolate"
        )

        best_model = polyLOC_interp(x_local)

        # extend model to full data array
        full_model = np.zeros_like(x)
        full_model[local_mask] = best_model

        # for points outside local window, extrapolate or use a constant
        if np.any(~local_mask):
            # Simple approach: use edge values
            full_model[x < x_local[0]] = best_model[0]
            full_model[x > x_local[-1]] = best_model[-1]
        
        # Detrend full light curve
        y_detrended = get_detrended_lc(y, full_model)

        # apply linear polynomial fit at the end
        
        # Apply linear correction (as in original)
        try:
            # Fit linear to out-of-transit detrended data
            linear_coeffs = np.polyfit(x[~mask], y_detrended[~mask], 1)
            linear_model = np.polyval(linear_coeffs, x)
            
            # Apply linear correction
            y_final_detrended = get_detrended_lc(y_detrended, linear_model)
            
        except:
            print("Linear correction failed, returning without linear detrending")
            y_final_detrended = y_detrended
            
    except Exception as e:
        print(f"Local detrending failed for this epoch: {e}")
        # Return NaN array if detrending fails
        y_final_detrended = np.full_like(y, np.nan)

    return y_final_detrended
