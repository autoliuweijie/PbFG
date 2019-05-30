"""
    Useful utils.
    @author: Liu Weijie
    @date: 2018-04-14
"""
from datetime import date, time
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np


def date_format_trans(in_date, outformat):
    """
        translate the input date to given format.
        outformat: 0 - datatime.date
                   1 - 'YYYY-MM-DD'
                   2 - 'YYYYMMDD'
    """
    unsupport_error = IOError(u"Error: This format of date is unsupport now!")

    # translate date to datatime.date
    if isinstance(in_date, str) or isinstance(in_date, int):

        in_date = str(in_date)

        if len(in_date) == 8:  # YYYYMMDD
            year, month, day = int(in_date[0: 4]), int(in_date[4: 6]), int(in_date[6: 8])
        elif len(in_date) == 10:  # YYYY-MM-DD
            year, month, day = int(in_date[0: 4]), int(in_date[5: 7]), int(in_date[8: 10])
        else:
            raise unsupport_error

        this_date = date(year=year, month=month, day=day)

    elif isinstance(in_date, date):

        this_date = in_date

    else:
        raise unsupport_error

    # translate datetime.date to outformat:
    if outformat == 0:
        return this_date
    elif outformat == 1:
        return this_date.strftime("%Y-%m-%d")
    elif outformat == 2:
        return this_date.strftime("%Y%m%d")
    else:
        raise IOError(u"Error: This outformat is unsupport now!")


def time_format_trans(in_time, outformat=0):
    """
        tranlate the in_time to given format.
        outformat: 0 - datatime.time
                   1 - HH:MM
    """

    unsupport_error = IOError(u"Error: This format of time is unsupport now!")

    if isinstance(in_time, str):

        if len(in_time) == 5:  # hh:mm
            hour, minute = int(in_time[0: 2]), int(in_time[3: 5])
        else:
            raise unsupport_error

        this_time = time(hour=hour, minute=minute)

    elif isinstance(in_time, time):

        this_time = in_time

    else:

        raise unsupport_error

    if outformat == 0:
        return this_time
    elif outformat == 1:
        return this_time.strftime("%H:%M")
    else:
        raise IOError(u"Error: This outformat is unsupport now!")


def show_signal(signal):
    plt.subplot(1, 2, 1)
    plt.plot(signal[0], signal[1], label='TI_940')
    plt.plot(signal[0], signal[2], label='TI_1450')
    plt.plot(signal[0], signal[3], label='TI_1550')
    plt.plot(signal[0], signal[4], label='TI_1710')
    plt.title("Transmitted Light Intensity Signals")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Temperature Signals")
    plt.plot(signal[0], signal[5], label="Tm")
    plt.plot(signal[0], signal[6], label="Tp")
    plt.legend(loc="upper left")


def show_signal_freqency(signal_freq, xlim=None):
    plt.subplot(1, 1, 1)
    for i in [1, 2, 3, 4]:
        plt.plot(signal_freq[0], signal_freq[i])
        if xlim: plt.xlim(xlim)
        plt.title("Frequency")


def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab


def corrcoef(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return num/den


def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == "__main__":
    print(date_format_trans('20180101', 1))


