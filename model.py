# Model of the MOS Transistor
# loads in reference data
# curve fits
# functions for plotting against the desired data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
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

VDSID = 0
VGSID = 1
VSBID = 2
IDSID = 3


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
        # subset = self.idvg_data[self.idvg_data[:, VSBID] == vsb_val].sort_values("VGS")
        subset = self.idvg_data[self.idvg_data[:, VSBID] == vsb_val]
        self.vgs = subset[:, VGSID]
        self.ids = subset[:, IDSID]
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
        return self.Kappa, self.Io
    
    def extract_all_kappas_IOs(self, plot=False):
        # Kappa should be fit for each curve
        kappas = []
        ios = []
        for vsb in (vsbs := np.unique(self.idvg_data[:, VSBID])):
            kappa, io = self.extract_kappa_I0(vsb)
            kappas.append(kappa)
            ios.append(io)
        if plot:
            plt.figure()
            plt.plot(vsbs, kappas)
            plt.title("K against VSB")
            plt.xlabel("VSB")
            plt.show()
            plt.figure()
            plt.plot(vsbs, ios)
            plt.title("I0 against VSB")
            plt.xlabel("VSB")
            plt.show()
        self.Kappa = kappas[0]
        self.Is = np.average(np.array(ios))
        
    def get_Vt(self, plot=False, vsb=0):
        # load data from VGS sweeps where VSB = -
        mask = (self.idvg_data[:, VSBID] == vsb) & (self.idvg_data[:, VDSID] == (np.max(self.idvg_data[:, VDSID])))
        VGS = self.idvg_data[:, VGSID][mask]
        ID = self.idvg_data[:, IDSID][mask]
        # take data close to intercept
        maxID = max(ID)
        minID = min(ID)
        diff = maxID - minID

        mask = (ID > 0.3*diff + minID) & (ID < 0.7*diff + minID)
        VGS_fit = VGS[mask]
        ID_fit = ID[mask]
        # linearize this line
        slope, intercept = np.polyfit(VGS_fit, ID_fit, 1)
        VGS_fit = np.linspace(0, 2.5, 100)
        ID_fit = slope * VGS_fit + intercept
        # print(f"slope: {slope}, intercept: {intercept}")
        # find index where ID = 0
        idx = np.where(ID_fit >= 0)[0][0]
        Vt = VGS_fit[idx]
        if plot:
            plt.figure()
            plt.plot(VGS, ID, label="data")
            plt.axvline(Vt, label="Vt0", color='red')
            plt.plot()
            plt.plot(VGS_fit, ID_fit, label='fitted data')
            plt.legend()
            plt.grid()
        return Vt
    
        print("FIT VT0 is CURRENTLY NOT IMPLEMENTED")

    def fit_Vts(self):
        # for now
        self.Vt0 = self.get_Vt(vsb=0)
        # eventually this should fit all VT parameters across voltages

    def fit_all(self):
        """
        Method to fit all parameters in order.
        """
        # generate kappas for each unique VSB
        self.extract_all_kappas_IOs() # this creates self.kappas
        self.fit_Vts()


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
        
        # forward current
        IF = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VSB)/(2*self.Ut)))**2
        # reverse current
        IR = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VDB)/(2*self.Ut)))**2
        # sum
        ID = IF - IR
        print(f"current {ID}")
        return ID
    
    def plot(self):
        """
        Plots model data against reference data
        """

        ############## PLOTTING ID VDS ###################
        unique_vgss = np.unique(self.idvd_data[:, VGSID])
        unique_vsbs = np.unique(self.idvd_data[:, VSBID])
        vdsmax = np.max(self.idvd_data[:, VDSID])
        vdmin = np.min(self.idvd_data[:, VDSID])
        vds_array = np.linspace(vdmin, vdsmax, 1000)
        
        fig, axs = plt.subplots(2, len(unique_vsbs), figsize=(15, 8))
        
        for i, vsb in enumerate(unique_vsbs):
            vdb_array = vds_array + vsb
            mask_vsb = self.idvd_data[:, VSBID] == vsb
            for vgs in unique_vgss:
                mask_vgs = self.idvd_data[:, VGSID] == vgs
                mask = mask_vgs & mask_vsb
                ##### plot reference data
                axs[0, i].plot(
                    self.idvd_data[mask][:, VDSID],
                    self.idvd_data[mask][:, IDSID],
                    label=f"Ref VGS: {vgs}",
                    linestyle = '--'
                )
                ###### plot model data
                axs[0, i].plot(
                    vds_array,
                    self.model(vgs + vsb, vsb, vdb_array),
                    label=f"VGS: {vgs}"
                )
            axs[0, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            axs[0, i].set_title(f"IDS / VDS Curves for VSB = {vsb}")

        ############# PLOTTING ID VGS ###################
        unique_vdss = np.unique(self.idvg_data[:, VDSID])
        unique_vsbs = np.unique(self.idvg_data[:, VSBID])
        vgsmax = np.max(self.idvg_data[:, VGSID])
        vgsmin = np.min(self.idvg_data[:, VGSID])
        vgs_array = np.linspace(vgsmin, vgsmax, 1000)

        for i, vds in enumerate(unique_vdss):
            mask_vds = self.idvg_data[:, VDSID] == vds
            for vsb in unique_vsbs:
                vgb_array = vgs_array + vsb
                mask_vsb = self.idvg_data[:, VSBID] == vsb
                mask = mask_vds & mask_vsb

                axs[1, i].plot(
                    self.idvg_data[mask][:, VGSID],
                    self.idvg_data[mask][:, IDSID],
                    label=f"Ref VDS: {vds}",
                    linestyle = '--'
                )
                ### model data ######
                axs[1, i].plot(
                    vgs_array,
                    self.model(vgb_array, vsb, vds + vsb),
                    label=f"VSB: {vsb}"
                )
            axs[1, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            axs[1, i].set_title(f"IDS / VGS Curves for VDS = {vds}")

        for ax in axs[0, :]:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
        for ax in axs[1, :]:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                )

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()


