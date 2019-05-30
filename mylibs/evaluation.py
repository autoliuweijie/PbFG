"""
    Some utils for evaluating model of noninvasive blood glucose estimation.
    @author: Liu Weijie
    @date: 2018-04-26
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calc_MAE(y0, y1):
    y0 = np.array(y0)
    y1 = np.array(y1)
    abs_error = np.abs(y0 - y1)
    mae = np.mean(abs_error)
    return mae


def calc_MRAE(y0, y1):
    y0 = np.array(y0)
    y1 = np.array(y1)
    relative_abs_error = np.abs(y0 - y1) / np.abs(y1)
    mrae = np.mean(relative_abs_error)
    return mrae


def calc_MSE(y0, y1):
    y0 = np.array(y0)
    y1 = np.array(y1)
    square_error = pow(y0 - y1, 2)
    mae = np.mean(square_error)
    return mae


class ClarkeErrorGrid(object):
    dot_style = "+"

    default_max_reference_axis = 28.0
    default_min_reference_axis = 0.0
    default_max_estimate_axis = 28.0
    default_min_estimate_axis = 0.0

    x_index = 0
    y_index = 1
    line_style = "k--"
    line_set = [
        [[0, 30], [0, 30]],
        [[3.89, 3.89], [10, 30]],
        [[0, 3.89], [10, 10]],
        [[3.89, 23.89], [10, 30]],
        [[0, 3.24], [3.89, 3.89]],
        [[3.24, 25], [3.89, 30]],
        [[3.89, 30], [2.48, 24]],
        [[3.89, 3.89], [0, 2.48]],
        [[7.22, 10], [0, 3.89]],
        [[10, 10], [0, 3.89]],
        [[10, 30], [3.89, 3.89]],
        [[3.89, 3.89], [4.668, 10]],
        [[14.3, 14.3], [3.89, 10]],
        [[14.3, 30], [10, 10]],
    ]

    region_label_coordinate = {
        (24, 21.5): "A",
        (21.5, 24): "A",
        (12, 6): "B",
        (6, 10): "B",
        (8.5, 0.5): "C",
        (10, 23): "C",
        (22, 7): "D",
        (2, 7): "D",
        (1.5, 20): "E",
        (20, 1.5): "E",

    }

    X_label = "Reference BGC (mmol/L)"
    Y_label = "Estimation BGC (mmol/L)"

    def __init__(self, reference_value=[], estimate_value=[]):
        self.reference_value = reference_value
        self.estimate_value = estimate_value
        self.set_axis(self.default_max_reference_axis, self.default_min_reference_axis,
                      self.default_max_estimate_axis, self.default_min_estimate_axis)

    def set_axis(self, reference_max, reference_min, estimate_max, estimate_min):
        self.max_reference_axis = reference_max
        self.min_reference_axis = reference_min
        self.max_estimate_axis = estimate_max
        self.min_estimate_axis = estimate_min

    def show(self, is_show=True, savepath=None):
        # preparing plt
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 1, 1)

        # set x, y range
        plt.xlim(self.min_reference_axis, self.max_reference_axis)
        plt.ylim(self.min_estimate_axis, self.max_estimate_axis)

        # draw region
        for line in self.line_set:
            plt.plot(line[self.x_index], line[self.y_index], self.line_style)

        # place region label
        for coordinate in self.region_label_coordinate:
            plt.text(coordinate[self.x_index], coordinate[self.y_index],
                     self.region_label_coordinate[coordinate], fontsize=28)

        # place axis label
        plt.xlabel(self.X_label, fontsize=18)
        plt.ylabel(self.Y_label, fontsize=18)

        # scatter
        plt.scatter(self.reference_value, self.estimate_value, marker=self.dot_style, c='b')

        if is_show:
            plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def calc_r2(self):
        coor = np.corrcoef(
            [
                self.reference_value,
                self.estimate_value,
            ]
        )
        return coor[0, 1] ** 2

    def calc_clarke(self):
        g_t = self.reference_value
        g_e = self.estimate_value

        num_A, num_B, num_C, num_D, num_E = 0, 0, 0, 0, 0
        num_data = len(g_t)
        for i in range(num_data):
            x, y = g_t[i], g_e[i]
            error = np.abs(x - y)
            error_rate = error / float(x)

            if error_rate <= 0.2:
                num_A += 1
            elif (y >= 3.89 and y <= 10 and x <= 3.89) or (y >= 3.89 and y <= 10 and x >= 14.3):
                num_D += 1
            elif (y >= 10 and x <= 3.89) or (y <= 3.89 and x >= 10):
                num_E += 1
            elif (y >= 10 and x >= 3.89 and (y - 10) >= (x - 3.89)) or (
                                    x <= 10 and x >= 7.22 and y <= 3.89 and y <= (x - 7.22)):
                num_C += 1
            else:
                num_B += 1

        A = float(num_A) / num_data
        B = float(num_B) / num_data
        C = float(num_C) / num_data
        D = float(num_D) / num_data
        E = float(num_E) / num_data

        return A + B, B, C, D, E


def show_test_result(y_true, y_pred, clarke=True):
    mae = calc_MAE(y_true, y_pred)
    mrae = calc_MRAE(y_true, y_pred)
    mse = calc_MSE(y_true, y_pred)
    myclark = ClarkeErrorGrid(y_true, y_pred)
    myclark.set_axis(28, 0, 28, 0)
    r2 = myclark.calc_r2()
    A_B, B, C, D, E = myclark.calc_clarke()

    print("=============Test Results===============")
    print("MAE: %s; MRAE: %s; MSE: %s; R2: %s;" % (mae, mrae, mse, r2))
    print("A+B: %s; B: %s; C: %s; D: %s; E: %s;" % (A_B, B, C, D, E))
    print("========================================")

    if clarke:
        myclark.show()


if __name__ == "__main__":
    myclark = ClarkeErrorGrid([1, 2, 3, 4, 5, 6, 7, 9], [1, 2, 3, 4, 5, 6, 7, 9])
    myclark.set_axis(28, 0, 28, 0)
    myclark.show()
    r2 = myclark.calc_r2()
    A_B, B, C, D, E = myclark.calc_clarke()