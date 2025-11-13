# Model of the MOS Transistor
# loads in reference data
# curve fits
# functions for plotting against the desired data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit


# Always true values
Kn = 450e-6
Na = 7e17 * 1e6 # cm^-3 to m^-3 
tox = 10.5e-9
ni = 1e10 * 1e6 # cm^-3
q = 1.602e-19 # C
Es = 11.7 * 8.854e-12 # F/m
Eox = 3.9 * 8.854e-12 # F/m
Cox = Eox/tox # F/m^2
k = 1.38e-23 # J/K
T = 300 # K
phiT = k*T/q
phiF = phiT * np.log(Na/ni)



class EKV_Model:
    def __init__(self, idvg_data : np.array, idvd_data : np.array, Width : float, Length : float):
        """
        EKV Model Class
        Initialize with inputs:
        idvg data - VG sweep data of IV, np array
        idvd data - VD sweep data of IV, np array
        Width - float width in m
        Length - float length in m
        """
        self.idvg_data = idvg_data
        self.idvd_data = idvd_data
        self.Width = Width
        self.Length = Length
        self.vds = None
        self.ids = None
        self.vgs = None
        self.vsb = None
        # all other EKV Model parameters here (incomplete)
        self.Is = None
        self.Io = None
        self.Kappa = None
        self.Vt0 = None
        self.mu_0 = None
        self.theta = None
        self.alpha = None
        self.phi_0 = None
        self.gamma = None
        self.Vfb = None
        self.tox = 10.5e-9
        self.e_ox = 3.45e-11
        self.Ut = 0.02585 # ~ .026
        self.cox = 3.45e-11 / 10.5e-9 # eox/toc
    # generic fitting function
    def fit_parameter(self):
        # 
        self.parameter = 0 # obviously edit this #
    
    def filter_data(self, datafile):
        '''Need to change this but a good starting point'''
        self.vds = datafile["VDS"].values
        self.ids = datafile["IDS"].values
        self.vgs = datafile["VGS"].values
        self.vsb = datafile["VSB"].values
    
    # kappa and Io extraction
    def extract_kappa_I0(self, vsb_val, window_size=7):
        '''This is created for finding all kappa and Io values for vsb value'''
        subset = self.idvg_data[self.idvg_data["VSB"] == vsb_val].sort_values("VGS")
        self.vgs = subset["VGS"].values
        self.ids = subset["IDS"].values
        ln_IDS = np.log(self.ids)
        best_r2 = -np.inf
        best_indices = None
        for i in range(len(self.vgs) - window_size):
            x_seg = self.vgs[i:i + window_size]
            y_seg = ln_IDS[i:i + window_size]
            slope, intercept, r_value, _, _ = linregress(x_seg, y_seg)
            if r_value**2 > best_r2:
                best_r2 = r_value**2
                best_indices = (i, i + window_size)

        i_start, i_end = best_indices
        x_lin = self.vgs[i_start:i_end]
        y_lin = ln_IDS[i_start:i_end]

        slope, intercept, r_value, _, _ = linregress(x_lin, y_lin)
        self.Kappa = slope * self.Ut
        self.Io = np.exp(intercept)
        return self.Kappa & self.Io

    def fit_all(self):
        """
        Method to fit all parameters in order.
        """
        raise NotImplementedError("fit_all is not complete yet")

    def model(self, VGB, VSB, VDB):
        """
        Runs the model. Uses fit values and EKV formula to return a drain current based on input voltages
        """
        if self.Kappa == None:
            raise ValueError("Fit Kappa before running model")
        if self.Is == None:
            raise ValueError("Fit Is before running model")
        if self.Vt0 == None:
            raise ValueError("Fit Vt0 before runnign model")
        if self.Ut == None:
            raise ValueError("Fit Ut before runnign model")
        
        # forward current
        IF = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VSB)/2*self.Ut))**2
        # reverse current
        IR = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VDB)/2*self.Ut))**2
        # sum
        ID = IF - IR
        return ID