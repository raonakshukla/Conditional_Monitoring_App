import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy import signal
import warnings as wr
from typing import Dict, List, Tuple, Optional
from scipy.signal import hilbert
from scipy.fft import rfft, rfftfreq
wr.filterwarnings('ignore')


# Load the dataset
df1 = pd.read_csv('Turbine_1.csv',low_memory=False)
df2 = pd.read_csv('Turbine_2.csv',low_memory=False)

# Changing the format for the time column
df1['t'] = pd.to_datetime(df1['t'], errors='coerce') # Converts the timestamps to Time Dela
df2['t'] = pd.to_datetime(df2['t'], errors='coerce') # Converts the timestamps to Time Dela

# Variables to be defined for spectral analysis
fs = 62.5                # Sampling Frequency 
hp_cut_hz = 0.1          # high-pass cutoff to remove gravity/drift. 
                     # It is assumed that the wind turbine is rotating at least 6 rpm(6/60 = 0.1Hz)
hp_order = 2
detrend = "linear" 
nperseg_sec = 20.0      # segment length (s)
noverlap_ratio = 0.5     # 50% overlap
window = "hann"
average = "mean"
scaling = "density"        # 'density' (PSD) or 'spectrum'

# Variable to be used for finding peak and modes of vibration
fmin_1p: float = 0.1            # Below 0.1Hz are already filtered using Buttress filter
fmax_1p: float = 2.0            # most rotors < 2 Hz 
min_prominence_db: float = 6.0  # prominence in dB for peak acceptance

# Mode search
fmin_mode: float = 0.2
fmax_mode: float = 31.0         # Nyquist ~31.25 at fs=62.5
max_modes: int = 4


# Function to extract time features from each window
def time_features(x: np.ndarray) -> dict:
    x = x.astype(float, copy=False)
    mean = x.mean()  # may indicate sensor bias or mounting issues
    rms  = np.sqrt((x**2).mean())  # overall vibration energy level
    std  = x.std(ddof=1) if len(x) > 1 else 0.0  # overall vibration energy level
    skew = (((x-mean)/(std+1e-12))**3).mean() if std > 0 else 0.0  #distribution
    kurt = (((x-mean)/(std+1e-12))**4).mean() if std > 0 else 3.0  # peakedness
    crest   = np.abs(x).max() / (rms + 1e-12)  # measures general impulsiveness
    impulse = np.abs(x).max() / (np.abs(x).mean() + 1e-12)  # more sensitive to occasional impulses
    shape   = rms / (np.abs(x).mean() + 1e-12)  # unbalance and misalignment defects
    absx = np.abs(x)
    max_abs = absx.max()
    clearance = max_abs / ( (np.mean(np.sqrt(absx)) + 1e-12) ** 2 )  # pick up local defects (like cracks, spalls, or pitting in bearings)
    margin    = max_abs / ( (np.mean(np.sqrt(np.sqrt(absx))) + 1e-12) ** 4 )  # more sensitive to weak impulsive activity than the clearance factor

    return dict(
        mean=float(mean), rms=float(rms), std=float(std), 
        skew=float(skew), kurtosis=float(kurt),
        crest_factor=float(crest), impulse_factor=float(impulse),
        shape_factor=float(shape),
        clearance_factor=float(clearance), margin_factor=float(margin)
    )

def spectral_features(x: np.ndarray, fs: float) -> dict:
    X = rfft(x)
    freqs = rfftfreq(len(x), 1/fs)
    mag = np.abs(X) / (len(x)/2)

    # dominant (ignore DC)
    idx = np.argmax(mag[1:]) + 1 if len(mag) > 1 else 0
    dom_f = float(freqs[idx]) if idx is not None else 0.0
    dom_a = float(mag[idx]) if idx is not None else 0.0

    # looseness
    env = np.abs(hilbert(x))
    env_kurt = (((env - env.mean())/(env.std(ddof=1)+1e-12))**4).mean() if len(env) > 1 else 0.0 

    return dict(dominant_freq=dom_f, dominant_amp=dom_a, envelope_kurtosis=float(env_kurt))


# Filter to remove effect of gravity. It is assumed that wind turbine is rotating at atleast 6 rpm
def butter_highpass(x: np.ndarray, fs: float, hp_cut_hz: float, hp_order: int = 2) -> np.ndarray:
    '''Buttress filter to remove the effect of gravity. Here, a high pass filter is used.'''
    if hp_cut_hz <= 0:
        return x.copy()
    b, a = signal.butter(hp_order, hp_cut_hz / (0.5 * fs), btype='highpass')
    return signal.filtfilt(b, a, x)

# Detrending the series
def preprocess_series(x: np.ndarray, type: str, fs: float, hp_cut_hz: float, hp_order: int) -> np.ndarray:
    ''' Function to detrend and remove high frequencies
    Input: An array of vibration signals from each channel
    Output: An array of filtered signal'''
    # Detrend first (helps with HPF transients)
    x_dt = signal.detrend(x, type = detrend) 
    return butter_highpass(x_dt, fs, hp_cut_hz, hp_order)

# Function Welch PSD 
def welch_psd(x: np.ndarray, fs: float, nperseg_sec: float, noverlap_ratio: float, 
              window: str, average: str, scaling: str) -> Tuple[np.ndarray, np.ndarray]:
    """ Welch PSD is used instead of FFT to  reduces the variance of the estimate"""
    nperseg = int(max(8, round(nperseg_sec * fs)))
    noverlap = int(round(noverlap_ratio * nperseg))
    f, Pxx = signal.welch(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        average=average, scaling=scaling, detrend = "constant"
    )
    return f, Pxx
    

# Function for Coherence Analysis
def coherence(x: np.ndarray, y: np.ndarray, fs: float, nperseg_sec: float, 
              noverlap_ratio: float, window: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates coherence between two signals"""
    nperseg = int(max(8, round(nperseg_sec * fs)))
    noverlap = int(round(noverlap_ratio * nperseg))
    f, Cxy = signal.coherence(x, y, fs=fs, window=window,
                              nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, Cxy

# Function for Magnitude to Decibel Conversion
def mag2db(x: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, eps))

# Function to find peaks
def find_peak_in_band(f: np.ndarray, pxx: np.ndarray, fmin: float, fmax: float,
                      min_prom_db: float = 6.0) -> Optional[Tuple[float, float]]:
    """Return (f_peak, amp_linear) or None if not found. The idea here is to find the peak of vibration.
    If the natural frequency of vibration decreases, it may be due to loss of stiffness due to failure
    of blade, erosion or increase in the weight of blade due to snow or water ingress"""
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return None
    fb, pb = f[band], pxx[band]
    # Use prominence on dB scale
    y = mag2db(pb)
    # Dynamic prominence threshold relative to local median
    baseline = np.median(y)
    prom = max(min_prom_db, 0.0)
    peaks, props = signal.find_peaks(y, prominence=prom)
    if peaks.size == 0:
        # Try relative threshold
        peaks, props = signal.find_peaks(y, height=baseline + prom)
        if peaks.size == 0:
            return None
    idx = peaks[np.argmax(y[peaks])]
    return float(fb[idx]), float(pb[idx])

# Function to find the peak modes to analyse the dominant mode of vibration
# Function to find the peak modes to analyse the dominant mode of vibration
def cross_spectral_matrix(X: np.ndarray, fs: float, nperseg_sec: float, noverlap_ratio: float = 0.5):
    """
    Compute frequency-dependent cross-spectral matrix S_xx(f) of shape (n_f, n_ch, n_ch).
    """
    nperseg = max(8, int(round(nperseg_sec*fs)))
    noverlap = int(round(noverlap_ratio*nperseg))
    f, Pxx = signal.welch(X, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                          detrend="constant", axis=0, scaling="density", return_onesided=True)
    # Use csd for off-diagonals
    n_f = len(f)
    n_ch = X.shape[1]
    S = np.zeros((n_f, n_ch, n_ch), dtype=complex)
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == j:
                S[:, i, j] = Pxx[:, i]
            else:
                f2, Pij = signal.csd(X[:, i], X[:, j], fs=fs, window="hann",
                                     nperseg=nperseg, noverlap=noverlap, detrend="constant",
                                     scaling="density")
                S[:, i, j] = Pij
                S[:, j, i] = np.conj(Pij)
    return f, S

def peak_pick_from_singular_values(f: np.ndarray, s1: np.ndarray, fmin: float, fmax: float, prom_db: float = 6.0, max_peaks: int = 6) -> List[float]:
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return []
    fb = f[band]
    y = mag2db(s1[band])
    peaks, props = signal.find_peaks(y, prominence=prom_db, distance=max(1, int(0.2 / (fb[1]-fb[0]))))
    if peaks.size == 0:
        return []
    prom = props.get("prominences", np.zeros_like(peaks))
    order = np.argsort(prom)[::-1]
    modes = [float(fb[p]) for p in peaks[order][:max_peaks]]
    modes.sort()
    return modes

# Function to find how much vibration is transferred from the tip to the roots
def transmissibility(pxx_tip: np.ndarray, pxx_root: np.ndarray) -> np.ndarray:
    """ Returns Vibration Transmissibility (or Frequency Response Ratio)
        Transmissibility = Output PSD ÷ Input PSD"""
    eps = 1e-20
    return (pxx_tip + eps) / (pxx_root + eps)


# simple linear interpolation
def interp(arrf, arrv, f0):
        if np.isnan(f0) or f0 < arrf[0] or f0 > arrf[-1]:
            return float("nan")
        return float(np.interp(f0, arrf, arrv))


st.set_page_config(page_title="Wind Sense", layout="wide")

st.title("Wind Sense ༄𖣘: Smart Condition Monitoring for Wind Turbines")


tab0, tab1, tab2, tab3, tab4 = st.tabs(["Time Domain", "Spectral Analysis", "Coherence Analysis", "Transmissibility", "Modal Analysis"])

with tab0:
    subtab = st.radio("Select View", ["Turbine 1", "Turbine 2"], key="subtab0")
    col = []
    axes = ["edge", "span", "flap"]
    blade = [1,2,3]
    for i in blade:
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{i}_{loc}_{ax}"
                col.append(name)
    if subtab == "Turbine 1":
        data_T1 = pd.DataFrame()

        for i in col:
            f1 = time_features(np.array(df1[i]))  # returns a dict of features
            f2 = spectral_features(np.array(df1[i]),fs=62.5)
            feats = f1|f2
            df_feat = pd.DataFrame.from_dict(feats, orient='index', columns=[i])
            data_T1 = pd.concat([data_T1, df_feat], axis=1)

        st.dataframe(data_T1,height=500)
    else: 
        data_T2 = pd.DataFrame()

        for i in col:
            f1 = time_features(np.array(df2[i]))  # returns a dict of features
            f2 = spectral_features(np.array(df2[i]),fs=62.5)
            feats = f1|f2
            df_feat = pd.DataFrame.from_dict(feats, orient='index', columns=[i])
            data_T2 = pd.concat([data_T2, df_feat], axis=1)

        st.dataframe(data_T2,height=500)
with tab1:
    # Build channel names
    psd_dict = {}
    axes = ["edge", "span", "flap"]
    blades = [1, 2, 3]

    blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_tab1")
    
    for loc in ["root", "tip"]:
        for ax in axes:
            name = f"B{blade}_{loc}_{ax}"
            z = df1[name].to_numpy(dtype=float)
            y = preprocess_series(z, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            f, Pxx = welch_psd(y, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window, average = average, scaling = scaling)
            z1 = df2[name].to_numpy(dtype=float)
            y1 = preprocess_series(z1, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            f1, Pxx1 = welch_psd(y1, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window, average = average, scaling = scaling)
            psd_dict[name] = [(f, Pxx), (f1, Pxx1)]

    # Welch PSD curves for D (two datasets overlaid per channel)
    max_plots_per_row = 3  # was ax_plots_per_row

    if psd_dict:
        n = len(psd_dict)
        ncols = min(max_plots_per_row, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            squeeze=False,
            sharex=False, sharey=False  # set True/True if you want common scales
        )
        flat_axes = axes.ravel()

        for i, (key, two_psds) in enumerate(psd_dict.items()):
            ax = flat_axes[i]
            # Unpack your two PSDs
            (f0, Pxx0), (f1, Pxx1) = two_psds

            ax.semilogy(f0, Pxx0, label="Turbine 1",alpha=1)
            ax.semilogy(f1, Pxx1, label="Turbine 2",alpha=0.8)
            ax.axvline(0.20833333, color="r", linestyle="--", linewidth=1, label="1P Line")


            ax.set_title(f"Welch PSD: {key}")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("PSD [units²/Hz]")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        # Hide any unused axes
        total = nrows * ncols
        for j in range(n, total):
            fig.delaxes(flat_axes[j])

        fig.tight_layout()
        st.pyplot(fig)

with tab2:
    subtab = st.radio("Select View", ["Coherence_PSD", "Coherence_at_Modes"], key="subtab2")
    if subtab == "Coherence_PSD":
    # Build channel names
        csd_dict = {}
        Coh_1p_T1 = {}
        Coh_1p_T2 = {}
        col = list()
        axes = ["edge", "span", "flap"]
        blades = [1, 2, 3]

        blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_tab2")
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                col.append(name)

        pairs = list(itertools.combinations(col, 2))
        for i, (col1, col2) in enumerate(pairs):
            z1 = df1[col1].to_numpy(dtype=float)
            x = preprocess_series(z1, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z2 = df1[col2].to_numpy(dtype=float)
            y = preprocess_series(z2, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z3 = df2[col1].to_numpy(dtype=float)
            x1 = preprocess_series(z3, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z4 = df2[col2].to_numpy(dtype=float)
            y1 = preprocess_series(z4, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            f, Cxy = coherence(x, y, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window)
            Coh_1p_T1[(col1,col2)] = interp(f,Cxy,0.20833333333333334)
            f1, Cxy1 = coherence(x1, y1, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window)
            Coh_1p_T2[(col1,col2)] = interp(f1,Cxy1,0.20833333333333334)
            csd_dict[(col1,col2)] =  [(f, Cxy), (f1, Cxy1)]
    # Welch PSD curves for D (two datasets overlaid per channel)
        max_plots_per_row = 3  # was ax_plots_per_row

        if csd_dict:
            n = len(csd_dict)
            ncols = min(max_plots_per_row, n)
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(5 * ncols, 3.5 * nrows),
                squeeze=False,
                sharex=False, sharey=False  # set True/True if you want common scales
            )
            flat_axes = axes.ravel()

            for i, (key, two_psds) in enumerate(csd_dict.items()):
                ax = flat_axes[i]
                # Unpack your two PSDs
                (f0, Cxy0), (f1, Cxy1) = two_psds

                ax.semilogx(f0, Cxy0, label="Turbine 1",alpha=1)
                ax.semilogx(f1, Cxy1, label="Turbine 2",alpha=0.8)
                ax.axvline(0.20833333, color="r", linestyle="--", linewidth=1, label="1P Line")

                ax.set_title(f"{key[0]} vs {key[1]}")
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel("Coherence")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")

            # Hide any unused axes
            total = nrows * ncols
            for j in range(n, total):
                fig.delaxes(flat_axes[j])

            fig.tight_layout()
            st.pyplot(fig)

    else:
        csd_dict = {}
        Coh_1p_T1 = {}
        Coh_1p_T2 = {}
        col = list()
        axes = ["edge", "span", "flap"]
        blades = [1, 2, 3]

        blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_sub_tab2")
        mode_map = {    '1P': 0.208333,
                        'Mode_1': 1.05,
                        'Mode_2': 4.3,
                        'Mode_3': 6.25,
                        'Mode_4': 8.2,
                        'Mode_5': 12.5,
                        'Mode_6': 16.5,
                        'Mode_7': 21.05,
                        'Mode_8': 24.85
                    }
        selection = st.radio(
                                "Modes",
                                options=list(mode_map.keys()),
                                horizontal=True,
                                format_func=lambda option: f"{option} ({mode_map[option]} Hz)"
                            )

        f0 = mode_map[selection]
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                col.append(name)

        pairs = list(itertools.combinations(col, 2))
        for i, (col1, col2) in enumerate(pairs):
            z1 = df1[col1].to_numpy(dtype=float)
            x = preprocess_series(z1, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z2 = df1[col2].to_numpy(dtype=float)
            y = preprocess_series(z2, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z3 = df2[col1].to_numpy(dtype=float)
            x1 = preprocess_series(z3, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            z4 = df2[col2].to_numpy(dtype=float)
            y1 = preprocess_series(z4, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
            f, Cxy = coherence(x, y, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window)
            Coh_1p_T1[(col1,col2)] = interp(f,Cxy,f0)
            f1, Cxy1 = coherence(x1, y1, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                            window = window)
            Coh_1p_T2[(col1,col2)] = interp(f1,Cxy1,f0)
            csd_dict[(col1,col2)] =  [(f, Cxy), (f1, Cxy1)]

        df_1 = pd.DataFrame.from_dict(Coh_1p_T1, orient='index',columns=['T1_Coh_1P'])
        df_2 = pd.DataFrame.from_dict(Coh_1p_T2, orient='index',columns=['T2_Coh_1P'])
        df = pd.concat([df_1,df_2],axis=1)

        x = np.arange(len(df))
        width = 0.35

        # Plot for increase in amplitudes at first harmonic
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot bars
        ax.bar(x - width/2, df["T1_Coh_1P"], width, label="Turbine 1")
        ax.bar(x + width/2, df["T2_Coh_1P"], width, label="Turbine 2")

        # Customize
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=90)
        ax.set_ylabel("Coherence")
        ax.set_title("T1 vs T2 Coherence at 1P")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

with tab3:
    subtab = st.radio("Select View", ["Transmissibilty_PSD", "Transmissibilty_at_Modes"], key="subtab3")
    if subtab == "Transmissibilty_PSD":
        psd_dict = {}
        axes = ["edge", "span", "flap"]
        blades = [1, 2, 3]

        blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_tab3")
        
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                z = df1[name].to_numpy(dtype=float)
                y = preprocess_series(z, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
                f, Pxx = welch_psd(y, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                                window = window, average = average, scaling = scaling)
                z1 = df2[name].to_numpy(dtype=float)
                y1 = preprocess_series(z1, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
                f1, Pxx1 = welch_psd(y1, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                                window = window, average = average, scaling = scaling)
                psd_dict[name] = [(f, Pxx), (f1, Pxx1)]
        trans_t1 = []
        t1_psd = []
        t2_psd = []
        trans_t2 = []
        trans_col = []
        col = list()
        axes = ["edge", "span", "flap"]
        fsd = psd_dict[f'B{blade}_root_edge'][0][0]
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                col.append(name)

        pairs = list(itertools.combinations(col, 2))
        for i in range(0, len(pairs)-1):
            for ax in axes:
                if ax in pairs[i][0] and ax in pairs[i][1]:
                    trans_col.append(pairs[i])
        for i in trans_col:
            t = transmissibility(psd_dict[i[1]][0][1],psd_dict[i[0]][0][1])
            t1_psd.append(t)
            inter = interp(fsd,t,0.20833333333333334)
            trans_t1.append(inter)
            t = transmissibility(psd_dict[i[1]][1][1],psd_dict[i[0]][1][1])
            t2_psd.append(t)
            inter = interp(fsd,t,0.20833333333333334)
            trans_t2.append(inter)
        
        # Comparison of Transmissibility for Turbine 1 and Turbine 2

        labels = ["edge", "span", "flap"]   # axis order
        indices = [0, 1, 2]                 # matching indices in t1_psd / t2_psd

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

        for ax, lab, idx in zip(axes, labels, indices):
            ax.semilogy(fsd, t1_psd[idx], label="Turbine 1")
            ax.semilogy(fsd, t2_psd[idx], alpha=0.8, label="Turbine 2")
            ax.axvline(0.20833333, color="r", linestyle="--", linewidth=1, label="1P Line")
            ax.set_title(f"{lab.capitalize()} Axis")
            ax.set_xlabel("Frequency [Hz]")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Transmissibility")
        axes[0].legend()

        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        psd_dict = {}
        axes = ["edge", "span", "flap"]
        blades = [1, 2, 3]

        blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_tab3")
        mode_map = {    '1P': 0.208333,
                        'Mode_1': 1.05,
                        'Mode_2': 4.3,
                        'Mode_3': 6.25,
                        'Mode_4': 8.2,
                        'Mode_5': 12.5,
                        'Mode_6': 16.5,
                        'Mode_7': 21.05,
                        'Mode_8': 24.85
                    }
        selection = st.radio(
                                "Modes",
                                options=list(mode_map.keys()),
                                horizontal=True,
                                format_func=lambda option: f"{option} ({mode_map[option]} Hz, key='trans')"
                            )

        f0 = mode_map[selection]        
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                z = df1[name].to_numpy(dtype=float)
                y = preprocess_series(z, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
                f, Pxx = welch_psd(y, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                                window = window, average = average, scaling = scaling)
                z1 = df2[name].to_numpy(dtype=float)
                y1 = preprocess_series(z1, fs=fs, hp_cut_hz= hp_cut_hz, hp_order=hp_order, type=detrend)
                f1, Pxx1 = welch_psd(y1, fs = fs, nperseg_sec = nperseg_sec, noverlap_ratio = noverlap_ratio,
                                window = window, average = average, scaling = scaling)
                psd_dict[name] = [(f, Pxx), (f1, Pxx1)]
        trans_t1 = []
        t1_psd = []
        t2_psd = []
        trans_t2 = []
        trans_col = []
        col = list()
        axes = ["edge", "span", "flap"]
        fsd = psd_dict[f'B{blade}_root_edge'][0][0]
        for loc in ["root", "tip"]:
            for ax in axes:
                name = f"B{blade}_{loc}_{ax}"
                col.append(name)

        pairs = list(itertools.combinations(col, 2))
        for i in range(0, len(pairs)-1):
            for ax in axes:
                if ax in pairs[i][0] and ax in pairs[i][1]:
                    trans_col.append(pairs[i])
        for i in trans_col:
            t = transmissibility(psd_dict[i[0]][0][1],psd_dict[i[1]][0][1])
            t1_psd.append(t)
            inter = interp(fsd,t,f0)
            trans_t1.append(inter)
            t = transmissibility(psd_dict[i[0]][1][1],psd_dict[i[1]][1][1])
            t2_psd.append(t)
            inter = interp(fsd,t,f0)
            trans_t2.append(inter)
        
        # Comparison of Transmissibility for Turbine 1 and Turbine 2
        labels = ["Edge", "Span", "Flap"]
        x = np.arange(len(labels))
        width = 0.35

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(6, 3.5))

        # Plot bars
        ax.bar(x - width/2, trans_t1, width, label="Turbine 1")
        ax.bar(x + width/2, trans_t2, width, label="Turbine 2")

        # Axis labeling and title
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Axis")
        ax.set_ylabel("Transmissibility at 1P")
        ax.set_title("Transmissibility Comparison at 1P")

        # Legend and grid
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Layout adjustment
        fig.tight_layout()

        # Display in Streamlit
        st.pyplot(fig,use_container_width=False)

with tab4:
    subtab = st.radio("Select View", ["Frequency Domain Decomposition", "Stochastic Subspace Identification"], key="subtab4")
    if subtab == "Frequency Domain Decomposition":
        mode = {}

        AXES = ["edge", "span", "flap"]
        LOCATIONS = ["root", "tip"]

        for blade in [1, 2, 3]:
            ch_names = []
            data_cols_1 = []
            data_cols_2 = []

            for loc in LOCATIONS:
                for ax in AXES:
                    name = f"B{blade}_{loc}_{ax}"
                    ch_names.append(name)
                    data_cols_1.append(df1[name].to_numpy(dtype=float))
                    data_cols_2.append(df2[name].to_numpy(dtype=float))

            X = np.vstack(data_cols_1).T   
            Y = np.vstack(data_cols_2).T   

            f_psd, S = cross_spectral_matrix(X, fs, nperseg_sec, noverlap_ratio)
            n_f = len(f_psd)
            s1 = np.zeros(n_f)

            for k in range(n_f):
                U, s, Vh = np.linalg.svd(S[k, :, :])
                s1[k] = s[0].real

            mode[f"T1_B{blade}"] = peak_pick_from_singular_values(
                f_psd, s1, fmin_mode, fmax_mode, min_prominence_db, max_peaks=8
            )

            f_psd, S = cross_spectral_matrix(Y, fs, nperseg_sec, noverlap_ratio)
            n_f = len(f_psd)
            s1 = np.zeros(n_f)

            for k in range(n_f):
                U, s, Vh = np.linalg.svd(S[k, :, :])
                s1[k] = s[0].real

            mode[f"T2_B{blade}"] = peak_pick_from_singular_values(
                f_psd, s1, fmin_mode, fmax_mode, min_prominence_db, max_peaks=8
            )

        df_mode = pd.DataFrame.from_dict(mode, orient="index",
                                        columns=['Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6','Mode 7','Mode 8'])
        df_mode.index.name = "Turbine_Blade"
        
        st.dataframe(df_mode,width=1000)
    else:
        blades = [1, 2, 3]

        blade = st.selectbox("Select Blade", blades, index=0, key=f"blade_selector_tab4")
        sheet_name = f"Mode_Shapes_B{blade}"
        df = pd.read_excel('mode_shapes.xlsx',sheet_name=sheet_name, na_filter=False)
        st.dataframe(df,width = 2000, hide_index = True)









