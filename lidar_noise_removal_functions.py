# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:19:33 2015

@author: jnewman
"""

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    #Adapted from code posted on Stack Overflow: http://stackoverflow.com/a/9815522
    #1-D linear interpolation to fill missing values
    
    #Inputs
    #A: Time series where NaNs need to be filled
    
    #Outputs
    #B: Time series with NaNs filled 
    
    from scipy import interpolate
    import numpy as np
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    #Only perform interpolation if more than 75% of the data are valid 
    if float(len(np.array(good).ravel()))/len(A) > 0.75:
        f = interpolate.interp1d(inds[good], A[good],bounds_error=False,fill_value='extrapolate')
        B = np.where(np.isfinite(A),A,f(inds))
    else:
        B = A
    return B
    
def spike_filter(ts,frequency):
    #Spike filter based on procedure used in Wang et al. (2015)
    #Reference: Wang, H., R. J. Barthelmie,  A. Clifton, and S. C. Pryor, 2015:
    #Wind measurements from arc scans with Doppler wind lidar, J. Atmos.
    #Ocean. Tech., 32, 2024–2040.

    
    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data 
    
    #Outputs
    #ts_filtered_interp: Filtered time series with NaNs filled in
    
    import numpy as np
    from lidar_noise_removal_functions import fill_nan
    

    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)

    ts_filtered = np.copy(ts)
    ts_filtered_interp = np.copy(ts)

    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        ts_window = ts_filtered[i:i+ten_min_count]
        
        #Calculate delta_v, difference between adjacent velocity values
        delta_v = np.zeros(len(ts_window)-1)
        for j in range(len(ts_window)-1):
            delta_v[j] = ts_window[j+1] - ts_window[j]
        q75, q25 = np.percentile(delta_v, [75 ,25])
        IQR= q75 - q25     
        #If abs(delta_v) at times i and i-1 are larger than twice the interquartile
        #range (IQR) and the delta_v values at times i and i-1 are of opposite sign, 
        #the velocity at time i is considered a spike and set to NaN.
        for j in range(1,len(ts_window)-1):
            if abs(delta_v[j]) > 2*IQR and abs(delta_v[j-1]) > 2*IQR:
                if np.sign(delta_v[j]) != np.sign(delta_v[j-1]):
                    ts_window[j] = np.nan
                    ts_filtered[i+j] = np.nan
        #Set entire 10-min. period to NaN if more than 40% of the velocity points 
        #are already NaN. 
        if (float(len(ts_window[np.isnan(ts_window)]))/len(ts_window)) > 0.4:
            ts_filtered[i:i+ten_min_count] = np.nan
        #Use a 1-D linear interpolation to fill in missing values
        ts_filtered_interp[i:i+ten_min_count] = fill_nan(ts_filtered[i:i+ten_min_count]) 
    return ts_filtered_interp
    
    
def acvf(ts):
    #Calculate autocovariance function for a time series

    #Inputs
    #ts: Time series of data

    #Outputs
    #ts_corr: Values of autovariance function starting from lag 0    
    
    import numpy as np
    lags = range(0,len(ts))
    ts_corr = []
    for i in lags:
        ts_subset_temp = ts[i:len(ts)]
        ts_subset_temp2 = ts[0:len(ts)-i]
        ts_corr.append(np.nanmean((ts_subset_temp-np.nanmean(ts_subset_temp))*(ts_subset_temp2-np.nanmean(ts_subset_temp2))))
    return ts_corr
    
def lenschow_technique(ts,frequency,mode_ws,option):
    #Apply different forms of the Lenschow et al. (2000) technique
    #Reference: Lenschow, D. H., V. Wulfmeyer, and C. Senff, 2000: Measuring second-through
    #fourth-order moments in noisy data. J. Atmos. Oceanic Technol., 17, 1330–1347.

    #Inputs
    #ts: Time series of data
    #frequency: Sampling frequency of data
    #mode_ws: raw_WC, VAD, or raw_ZephIR 
    #mode_noise: Type of Lenschow noise correction to be applied. Options are linear, subrange, and spectrum.

    #Outputs
    #new_ts_var: 10-min. variance after noise correction has been applied 

    import numpy as np
    from scipy.optimize import curve_fit
    from lidar_noise_removal_functions import acvf, inertial_subrange_func, fill_nan
    from lidar_preprocessing_functions import get_10min_var
    ts = fill_nan(ts)    
    
    #Number of samples in a 10-min period
    ten_min_count = int(frequency*60*10)
    var_diff = []
    var_orig = []
    lags = np.arange(0,ten_min_count)/float(frequency)
    for i in np.arange(0,len(ts)-ten_min_count+1,ten_min_count):
        #10-min. window of data
        ts_window = ts[i:i+ten_min_count]
        ten_min_index = (i-1)/ten_min_count + 1
        var_orig.append(get_10min_var(ts_window,frequency))
        if 'linear' in option:
            #Use values of ACVF from first four non-zero lags to linearly extrpolate
            #ACVF to lag 0
            ts_corr = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_corr[1:4]
            p = np.polyfit(x_vals,y_vals,1)
            var_diff.append(var_orig[ten_min_index]-p[1])
        if 'subrange' in option:
            #Use values of ACVF from first four non-zero lags to produce fit 
            #to inertial subrange function. Value of function at lag 0 is assumed
            #to be the true variance. 
            ts_corr = acvf(ts_window)
            x_vals = lags[1:4];
            y_vals = ts_corr[1:4]
            try:
                popt, pcov = curve_fit(inertial_subrange_func, x_vals, y_vals,\
                p0 = [np.mean((ts_window-np.mean(ts_window))**2),0.002])
                var_diff.append(var_orig[ten_min_index]-popt[0])   
            except:
                 var_diff.append(np.nan)   
        if 'spectrum' in option:
           #Assume spectral power at high frequencies is due only to noise. Average
           #the spectral power at the highest 20% of frequencies in the time series
           #and integrate the average power across all frequencies to estimate the
           #noise floor 
           from lidar_preprocessing_functions import get_10min_spectrum, get_10min_spectrum_WC_raw
           import numpy.ma as ma
           if "raw_WC" in mode_ws:
               [S,fr] = get_10min_spectrum_WC_raw(ts_window,frequency)
           else:
               [S,fr] = get_10min_spectrum(ts_window,frequency)
           x = ma.masked_inside(fr,0.8*fr[-1],fr[-1])
           func_temp = []
           for j in range(len(fr)):
               func_temp.append(np.mean(S[x.mask]))
           noise_floor = np.trapz(func_temp,fr)
           var_diff.append(noise_floor)
    
    var_diff = np.array(var_diff)
    #Only use var_diff values where the noise variance is positive
    var_diff[var_diff < 0] = 0    
    new_ts_var = np.array(var_orig)-var_diff
    return new_ts_var 
    
def inertial_subrange_func(t, b, C):
    #Inertial subrange fit for autocovariance function  
    #t is lag time, b is the variance at lag 0, and C is a parameter corresponding to eddy dissipation
    
    return -C*t**(2./3) + b    