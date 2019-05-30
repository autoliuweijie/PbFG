# encoding: utf-8
"""
    There are some utils used for processing dataset.
    @author: Liu Weijie
    @date: 2018-10-29
"""
import os
import numpy as np
import pandas as pd
from mylibs import utils
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
from mylibs.signal_preprocess import cut_head


def parse_record_signal_file(file_url, with_signal=True, Ts = 40):

    signal_df = pd.read_csv(file_url)

    # fetch bgc and r_time
    bgc = float(signal_df.iloc[0]['bgc'])
    r_time = utils.time_format_trans(signal_df.iloc[0]['time'], 0)

    signal = None
    if with_signal:
        # fetch transmission light signal
        TI_940 = signal_df['940_TI'].values
        TI_1450 = signal_df['1450_TI'].values
        TI_1550 = signal_df['1550_TI'].values
        TI_1710 = signal_df['1710_TI'].values
        Tm = signal_df['Tm'].values
        Tp = signal_df['Tp'].values
        Times = np.arange(1, len(TI_940) + 1) * 40

        signal = np.array([Times, TI_940, TI_1450, TI_1550, TI_1710, Tm, Tp])

    return bgc, r_time, signal


def read_navy_origin_data(dataset_path):
    """
        Read origin singnal data from dataset_path
        return: p_infos_df - pd.DataFrame
                records_meta_df - pd.DataFrame
    """
    # read navy_patients_infos.csv to pd.DataFrame
    p_infos_path = os.path.join(dataset_path, 'navy_patients_infos.csv')
    p_infos_df = pd.read_csv(p_infos_path)

    # read navy_dataset records to pd.DaraFrame and np.narray
    records_meta_dict = {
        'p_id(i)': [],
        'date(d)': [],
        'time(t)': [],
        'BGC(f)': [],
        'signal_url(s)': []
    }
    for idx in range(p_infos_df.shape[0]):  # read data per patient

        # get patient infos
        p_id = p_infos_df.iloc[idx]['p_id(i)']
        p_name = p_infos_df.iloc[idx]['p_name(s)']
        in_date = utils.date_format_trans(str(p_infos_df.iloc[idx]['in_date(d)']), 0)
        out_date = utils.date_format_trans(str(p_infos_df.iloc[idx]['out_date(d)']), 0)

        today = in_date
        while today <= out_date:  # read data per date

            data_dirpath = os.path.join(
                dataset_path,
                'navy_dataset',
                '%s_%s' % (p_id, p_name),
                utils.date_format_trans(today, 1)
            )

            if os.path.exists(data_dirpath):

                record_files = os.listdir(data_dirpath)
                for record_file in record_files:

                    if record_file.split('.')[-1] != 'csv' or record_file[0] == '.':
                        continue

                    signal_url = os.path.join(data_dirpath, record_file)
                    bgc, r_time, _ = parse_record_signal_file(signal_url, with_signal=False)

                    # intergate data
                    records_meta_dict['p_id(i)'].append(p_id)
                    records_meta_dict['date(d)'].append(today)
                    records_meta_dict['time(t)'].append(r_time)
                    records_meta_dict['BGC(f)'].append(bgc)
                    records_meta_dict['signal_url(s)'].append(signal_url)

            today += timedelta(days=1)

    records_meta_df = pd.DataFrame(records_meta_dict)

    return p_infos_df, records_meta_df


class BGCSignalDataset(Dataset):

    def __init__(self, pmeta_frame, transform=None):
        self.pmeta_frame = pmeta_frame
        self.transform = transform

    def __len__(self):
        return len(self.pmeta_frame)

    def __getitem__(self, idx):
        idx = int(idx)
        bgc, r_time, signal = parse_record_signal_file(self.pmeta_frame.iloc[idx]['signal_url(s)'])
        signal_cut, incident_intensity = cut_head(signal)
        pid = self.pmeta_frame.iloc[idx]['p_id(i)']

        sample = {
            'bgc': bgc,
            'signal': signal_cut,
            'pid': pid,
            'Ii': incident_intensity,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    navy_path = '/root/hdd/datasets/navy_glucose'
    p_infos_df, records_meta_df = read_navy_origin_data(navy_path)
    print(records_meta_df.head())
