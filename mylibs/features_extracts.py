# coding: utf-8
"""
    some function for extracting features.

    @author: Liu Weijie
"""
from mylibs.signal_preprocess import get_pds_region_edage, get_pws_timeset
from mylibs.signal_preprocess import PDS_signal_extract, PWS_signal_extract
from scipy.optimize import curve_fit
import math
import numpy as np
import copy


def get_pds_features(pds, incident_intensity):
    s_gap, l_gap, h_gap, e_gap = get_pds_region_edage(pds)

    # calculate mLpA
    mLpA = []
    for i in [1, 2, 3, 4]:
        I_i = incident_intensity[i]
        LpA = []
        for I_t in pds[i, s_gap: l_gap]:
            LpA.append(math.log(I_i / I_t))
        mLpA.append(LpA)
    mLpA = np.mean(mLpA, axis=1)
    mLpA[mLpA < 0] = 0

    # calculate mHpA
    mHpA = []
    for i in [1, 2, 3, 4]:
        I_i = incident_intensity[i]
        HpA = []
        for I_t in pds[i, h_gap: e_gap]:
            HpA.append(math.log(I_i / I_t))
        mHpA.append(HpA)
    mHpA = np.mean(mHpA, axis=1)
    mHpA[mHpA < 0] = 0

    # calculate MpA
    MpA = []
    for i in [1, 2, 3, 4]:
        I_i = incident_intensity[i]
        MpA.append(math.log(I_i / pds[i, int((l_gap + h_gap) / 2)]))
    MpA = np.array(MpA)
    MpA[MpA < 0] = 0

    # calculate mRpA
    mRpA = []
    for i in [1, 2, 3, 4]:
        RpA = []
        for It_h in pds[i, h_gap: e_gap]:
            for It_l in pds[i, s_gap: l_gap]:
                RpA.append(math.log(It_h / It_l))
        mRpA.append(RpA)
    mRpA = np.mean(mRpA, axis=1)
    mRpA[mRpA < 0] = 0

    features = {'mLpA': mLpA, 'mHpA': mHpA, 'MpA': MpA, 'mRpA': mRpA}

    return features


def get_pws_features(pws, incident_intensity):
    te_idx, tc_idx = get_pws_timeset(pws)
    match_num = min(len(te_idx), len(tc_idx))

    # calculate mEvA
    mEvA = []
    for i in [1, 2, 3, 4]:
        I_i = incident_intensity[i]
        EvA = []
        for j in range(0, match_num):
            EvA.append(math.log(I_i / pws[i, te_idx[j]]))
        mEvA.append(EvA)
    mEvA = np.mean(mEvA, axis=1)
    mEvA[mEvA < 0] = 0

    # calculate mCvA
    mCvA = []
    for i in [1, 2, 3, 4]:
        I_i = incident_intensity[i]
        CvA = []
        for j in range(0, match_num):
            CvA.append(math.log(I_i / pws[i, tc_idx[j]]))
        mCvA.append(CvA)
    mCvA = np.mean(mCvA, axis=1)
    mCvA[mCvA < 0] = 0

    # calculate mRvA
    mRvA = []
    for i in [1, 2, 3, 4]:
        RvA = []
        for j in range(0, match_num):
            RvA.append(math.log(pws[i, tc_idx[j]] / pws[i, te_idx[j]]))
        mRvA.append(RvA)
    mRvA = np.mean(mRvA, axis=1)
    mRvA[mRvA < 0] = 0

    # calculate ma
    ma = []
    for i in [1, 2, 3, 4]:
        ma_tmp = [abs(pws[i, te_idx[j]] - pws[i, tc_idx[j]]) for j in range(0, match_num)]
        ma.append(ma_tmp)
    ma = np.mean(ma, axis=1)

    # calculate mf
    mf = np.mean((1000 / (np.abs(np.array(tc_idx) - np.array(te_idx)) * 40)))

    features = {'mEvA': mEvA, 'mCvA': mCvA, 'mRvA': mRvA, 'ma': ma, 'mf': mf}

    return features


def exponent(x, k, tao, beta):
    y = k * np.exp(-tao * x) + beta
    return y


def average_filter(signal, window_size=5):
    new_signal = [signal[0]]
    for i in range(1, len(signal)):
        from_idx = (i - window_size) if (i - window_size) >= 0 else 0
        ave = np.mean(signal[from_idx: i])
        new_signal.append(ave)

    return new_signal


def get_temperature_features(signal, ave=5):

    try:
        Tm, Tp = copy.deepcopy(signal[5]), copy.deepcopy(signal[6])
        Time = copy.deepcopy(signal[0])

        start_idx = int(5000 / 40)
        k_c = -1  # 为了拟合，需要对原始数据做一些变换, 转换为向下收敛的更好计算
        beta_c = 28
        tao_c = 1000
        Tm_c = k_c * np.array(average_filter(Tm, window_size=ave)) + beta_c
        Tp_c = k_c * np.array(average_filter(Tp, window_size=ave)) + beta_c
        Time_c = Time / tao_c

        # 指数拟合Tp
        popt, pcov = curve_fit(exponent, Time_c, Tp_c, maxfev=100000)
        k_p, tao_p, beta_p = k_c * popt[0], popt[1] / tao_c, (beta_c - popt[2])

        # 指数拟合Tm
        popt, pcov = curve_fit(exponent, Time_c, Tm_c, maxfev=100000)
        k_m, tao_m, beta_m = k_c * popt[0], popt[1] / tao_c, (beta_c - popt[2])

        # 求解交叉点
        distance = []
        t0 = 0
        potential_t = []
        for i in range(2000):
            potential_t.append(t0)
            Tp_0 = exponent(t0, k_p, tao_p, beta_p)
            Tm_0 = exponent(t0, k_m, tao_m, beta_m)
            dis = abs(Tp_0 - Tm_0)
            if Tp_0 > Tm_0:
                t0 -= 40  # 减去 40ms
            if dis <= 0.1:
                break
            distance.append(dis)
        # time_init = potential_t[np.argmin(distance)]
        temperature_init = (exponent(t0, k_p, tao_p, beta_p) + exponent(t0, k_m, tao_m, beta_m)) / 2.0

        if temperature_init >= 39:
            temperature_init = 39.0
        if temperature_init < 25:
            temperature_init = 25

        # 组成features
        features = {
            "Tm_init": temperature_init,
            "Tear": beta_p,
            "Tao_p": tao_p
        }

        # coefficient
        coefficients = {
            "k_p": k_p, "tao_p": tao_p, "beta_p": beta_p,
            "k_m": k_m, "tao_m": tao_m, "beta_m": beta_m,
        }
    except:
        features = {'Tm_init': 25, 'Tear': 34.148632467453844, 'Tao_p': 5.105899211796868e-05}
        coefficients = {
            'k_p': -12.334635302183296,
            'tao_p': 5.105899211796868e-05,
            'beta_p': 34.148632467453844,
            'k_m': -16878.101968175943,
            'tao_m': 8.200575853826243e-09,
            'beta_m': 16897.5049582104
        }

    return features, coefficients


def get_pws_shape_features(pws):
    # Intercept a signal of one cycle
    te_idx, tc_idx = get_pws_timeset(pws)
    if len(te_idx) >= 2:
        start_idx = te_idx[0]
        end_idx = te_idx[1]
    else:
        start_idx = 0
        end_idx = len(pws[1])
    time_cycle = pws[0][start_idx: end_idx]
    Tt_cycle = pws[1][start_idx: end_idx]  # 940nm is best

    # normalize cycle signal
    max_v, min_v, length = max(Tt_cycle), min(Tt_cycle), len(Tt_cycle)
    Tt_cycle_nor = []
    len_nor = 100  # 标准化后的长度
    for i in range(len_nor):
        this_value = Tt_cycle[int((i / len_nor) * length)]
        this_value = (this_value - min_v) / (max_v - min_v)
        Tt_cycle_nor.append(this_value)
    Tt_cycle_nor = np.array(Tt_cycle_nor)

    # calculate features
    shape_feature = []
    feature_range = [[0, 19], [20, 39], [40, 59], [60, 79], [80, 100]]
    for r in feature_range:
        shape_feature.append(np.mean(Tt_cycle_nor[r[0]: r[1]]))
    shape_feature = np.array(shape_feature)

    cycle = {
        'Tt_cycle': Tt_cycle,
        'Tt_cycle_nor': Tt_cycle_nor,
    }

    return shape_feature, cycle


def signal2features(signal, Ii, shape_features=False):
    signal = np.array(signal)
    Ii = np.array(Ii)

    pds, _ = PDS_signal_extract(signal)
    pws, _ = PWS_signal_extract(signal, AC=[1, 1.5])
    pws_features = get_pws_features(pws, Ii)
    pds_features = get_pds_features(pds, Ii)
    tem_features, _ = get_temperature_features(signal)

    x = []
    all_features = {}
    all_features.update(pws_features)
    all_features.update(pds_features)
    all_features.update(tem_features)
    for k, v in all_features.items():
        if isinstance(v, (int, float)):
            x.append(v)
        else:
            x += v.tolist()
    if not shape_features:
        return np.array(x)
    else:
        shape_features, _ = get_pws_shape_features(pws)
        return np.array(x), shape_features


