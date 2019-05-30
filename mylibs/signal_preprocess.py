# coding: utf-8
"""
    Signal preprocess function

    @author: Liu Weijie
    @date: 2018-10-31
"""
import numpy as np
from scipy.fftpack import fft, ifft
import copy
from sklearn.cluster import KMeans


def cut_head(signal, thresh=20000):
    useful_start = np.max(np.nonzero(signal[1:5] > thresh)[1]) + 5
    useful_signal = signal[:, useful_start:]

    incident_intensity = np.mean(signal[:, :useful_start], axis=1)
    return useful_signal, incident_intensity


def conduct_fft(signal):
    Ts = signal[0][1] - signal[0][0]  # sample interval
    num_samples = signal.shape[1]
    base_freq = (1000.0 / Ts) / num_samples
    fs = np.arange(1, num_samples + 1) * base_freq  # get frquency axis

    signal_freq = [fs, ]
    for i in [1, 2, 3, 4]:  # for Ti signal
        tmp_freq = fft(signal[i])
        signal_freq.append(tmp_freq)

    signal_freq.append(signal[5])  # for temperature signal
    signal_freq.append(signal[6])

    return np.array(signal_freq)


def conduct_ifft(signal_freq, start_ts=0):
    base_freq = signal_freq[0][1] - signal_freq[0][0]
    num_samples = signal_freq.shape[1]
    Ts = (1000 / (base_freq * num_samples)).real
    ts = np.arange(1, num_samples + 1) * Ts + start_ts

    signal_time = [ts, ]
    for i in [1, 2, 3, 4]:
        tmp_time = ifft(signal_freq[i]).real
        signal_time.append(tmp_time)

    signal_time.append(signal_freq[5])
    signal_time.append(signal_freq[6])

    return np.array(signal_time)


def PDS_signal_extract(signal, DC=[0.01, 0.5]):

    start_ts = signal[0, 0]
    signal_freq = conduct_fft(signal)
    signal_freq_for_ret = np.real(copy.deepcopy(signal_freq))
    for i in [1, 2, 3, 4]:
        signal_freq[i][signal_freq[0] < DC[0]] = 0
        signal_freq[i][signal_freq[0] > DC[1]] = 0
    signal_back = conduct_ifft(signal_freq, start_ts)
    return np.real(signal_back), signal_freq_for_ret


def get_pds_region_edage(signal_pds, start_rate=0.1, end_rate=0.9):
    gaps = []
    for i in [1, 2, 3, 4]:

        signal = signal_pds[i]
        num_samples = len(signal)
        start = int(start_rate * num_samples)
        end = int(end_rate * num_samples)
        signal_cut = copy.deepcopy(signal[start: end])

        # extract features
        features = []
        length = len(signal_cut)
        for j in range(length):
            value_f = signal_cut[j]
            slope_f = 0
            if 2 <= j < length - 2:
                slope_f = signal_cut[j + 1] + signal_cut[j + 2] - signal_cut[j - 1] - signal_cut[j - 2]
            elif j < 2:
                slope_f = 2 * (signal_cut[j + 1] + signal_cut[j + 2] - 2 * signal_cut[j])
            else:
                slope_f = 2 * (2 * signal_cut[j] - signal_cut[j - 1] - signal_cut[j - 2])
            features.append([value_f, slope_f])
        features = np.real(np.array(features))
        features_nor = (features - np.mean(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

        # k-means
        kmeans = KMeans(n_clusters=3, random_state=0).fit(features_nor)
        labels = kmeans.labels_

        gap = np.nonzero(np.array([labels[k + 1] - labels[k] for k in range(0, len(labels) - 1)]))[0][0: 2]
        gaps.append([start, start + gap[0], start + gap[1] + 0.2 * (end - gap[1]), end])

    gaps = np.array(gaps)

    s_gap = int(np.max(gaps[:, 0]))
    l_gap = int(np.min(gaps[:, 1]))
    h_gap = int(np.max(gaps[:, 2]))
    e_gap = int(np.min(gaps[:, 3]))

    return s_gap, l_gap, h_gap, e_gap


def PWS_signal_extract(signal, AC=(1, 1.5)):
    # get low phase region from pds
    signal_pds, _ = PDS_signal_extract(signal, DC=[0.01, 0.5])
    idx_low_start, idx_low_end, _, _ = get_pds_region_edage(signal_pds)

    # get lowphase signal
    signal_lowphase = signal[:, idx_low_start: idx_low_end]
    start_ts = signal_lowphase[0, 0]

    # filtering
    signal_lowphase_freq = conduct_fft(signal_lowphase)
    signal_lowphase_freq_for_ret = np.real(copy.deepcopy(signal_lowphase_freq))
    for i in [1, 2, 3, 4]:
        signal_lowphase_freq[i][signal_lowphase_freq[0] < AC[0]] = 0
        signal_lowphase_freq[i][signal_lowphase_freq[0] > AC[1]] = 0
    signal_lowphase_back = conduct_ifft(signal_lowphase_freq, start_ts)

    # add pds in low phase
    signal_pws = signal_lowphase_back + signal_pds[:, idx_low_start: idx_low_end]
    signal_pws[0] = signal_pws[0] / 2  # 防止时间和温度被翻倍
    signal_pws[5] = signal_pws[5] / 2
    signal_pws[6] = signal_pws[6] / 2

    return np.real(signal_pws), signal_lowphase_freq_for_ret


def get_pws_timeset(signal_pws, time_window=1000):
    ts = (signal_pws[0, -1] - signal_pws[0, 0]) / len(signal_pws[0])
    window_size = int(time_window / ts) + 1

    te_idx, tc_idx = [], []
    for i in range(window_size, len(signal_pws[0]) - window_size):
        this_value = signal_pws[1][i]
        near_max = np.max(signal_pws[1][i - window_size: i + window_size])
        near_min = np.min(signal_pws[1][i - window_size: i + window_size])

        if this_value == near_max and (len(tc_idx) <= len(te_idx)):
            tc_idx.append(i)
        if this_value == near_min and (len(te_idx) <= len(tc_idx)):
            te_idx.append(i)

    match_num = min(len(te_idx), len(tc_idx))
    te_idx, tc_idx = te_idx[: match_num], tc_idx[: match_num]

    return te_idx, tc_idx