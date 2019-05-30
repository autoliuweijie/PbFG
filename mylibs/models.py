# coding: utf-8
"""
    Some models
    @author: Liu Weijie
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from mylibs.utils import corrcoef, sigmoid
from mylibs.features_extracts import signal2features
import pandas as pd
import numpy as np
import copy


class PWSClassification(object):
    """PWS分类的抽象类"""

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class KnnPwsClassification(PWSClassification):
    """基于KNN的脉搏波分类"""

    def __init__(self, n_neighbors):
        super(KnnPwsClassification, self).__init__()
        self.n_neighbors = n_neighbors
        self.kernel = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
        )

    def fit(self, features_file):
        features_type_df = pd.read_csv(features_file)
        X = features_type_df[['f_0', 'f_1', 'f_2', 'f_3', 'f_4']].values
        y = features_type_df['type'].values
        self.kernel.fit(X, y)
        print('X shape: ', X.shape)
        print('y shape: ', y.shape)

    def predict(self, features_batch):
        return self.kernel.predict(features_batch)


class IBGCEstimator(object):
    """用于血糖估计的抽象类"""

    def __init__(self):
        pass

    def fit(self,
            X,
            y):
        pass

    def predict(self,
                X):
        pass


class SKLBGCEstimator(IBGCEstimator):
    """从Sklearn适配的血糖估计器抽象类"""

    def __init__(self):
        super(SKLBGCEstimator, self).__init__()
        self.kernel = None
        self.X_mean = None
        self.X_max = None
        self.X_min = None

    def fit(self,
            X_train,
            Y_train,
            X_val=None,
            Y_val=None,
            num_epoch=1,
            verbose=True):

        # normlize X
        self.X_mean = np.mean(X_train, axis=0)
        self.X_max = np.max(X_train, axis=0)
        self.X_min = np.min(X_train, axis=0)
        X_train = self._normalize_X(X_train)
        X_val = self._normalize_X(X_val) if X_val is not None else None

        # start training
        r_2 = 0
        if verbose: print("Start training...")
        for i_epoch in range(1, num_epoch + 1):

            X_train, Y_train = self._shuffle_XY(X_train, Y_train)
            if verbose: print("Epoch %s / %s." % (i_epoch, num_epoch))

            self.kernel.fit(X_train, Y_train)

            # validation
            if X_val is None or Y_val is None: continue
            Y_pre = self.kernel.predict(X_val).reshape((-1,))
            Y_pre = self._limit_value(Y_pre)
            corr = corrcoef(Y_val, Y_pre)
            r_2 = corr ** 2
            if verbose: print("R^2 = %s" % (r_2))

        if verbose: print("Finish training")

        return r_2

    def predict(self,
                X):

        X_nor = self._normalize_X(X)
        Y_pre = self.kernel.predict(X_nor).reshape((-1,))
        Y_pre = self._limit_value(Y_pre)
        return Y_pre

    def _limit_value(self, y):
        y[y >= 28] = 28
        y[y <= 3.2] = 3.2
        return y

    def _normalize_X(self, X):
        X_nor = (X - self.X_mean) / (self.X_max - self.X_min + 1e-7)

        return X_nor

    def _shuffle_XY(self,
                    X,
                    Y):
        re_idx = np.random.permutation(len(Y))
        new_X = X[re_idx, :]
        new_Y = Y[re_idx]
        return new_X, new_Y


class PLSEstimator(SKLBGCEstimator):

    def __init__(self,
                 n_components=2,
                 max_iter=500):

        super(PLSEstimator, self).__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.kernel = PLSRegression(n_components=self.n_components,
                                    scale=False,
                                    max_iter=self.max_iter)


class SVREstimator(SKLBGCEstimator):

    def __init__(self,
                 kernel='rbf',
                 gamma='auto',
                 C=1.0,
                 epsilon=0.2):

        super(SVREstimator, self).__init__()
        self.kernel_svm = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon=epsilon
        self.kernel = SVR(kernel=self.kernel_svm,
                          gamma=self.gamma,
                          C=self.C,
                          epsilon=self.epsilon)


class NNEstimator(SKLBGCEstimator):

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation='relu',
                 solver='adam',
                 alpha=0.0001,  # L2
                 batch_size='auto',
                 learning_rate='constant',
                 learning_rate_init=0.001):

        super(NNEstimator, self).__init__()
        self.kernel = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,  # L2
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init
        )

    def fit(self,
            X_train,
            Y_train,
            X_val=None,
            Y_val=None,
            num_epoch=100,
            verbose=True):

        # normlize X
        self.X_mean = np.mean(X_train, axis=0)
        self.X_max = np.max(X_train, axis=0)
        self.X_min = np.min(X_train, axis=0)
        X_train = self._normalize_X(X_train)
        X_val = self._normalize_X(X_val) if X_val is not None else None

        # start training
        r_2 = 0
        if verbose: print("Start training...")
        for i_epoch in range(1, num_epoch + 1):

            X_train, Y_train = self._shuffle_XY(X_train, Y_train)
            if verbose: print("Epoch %s / %s." % (i_epoch, num_epoch))

            self.kernel.partial_fit(X_train, Y_train)  # 只训练一个Batch，且保证下次再训练不会重新初始化

            # validation
            if X_val is None or Y_val is None: continue
            Y_pre = self.kernel.predict(X_val).reshape((-1,))
            Y_pre = self._limit_value(Y_pre)
            try:
                corr = corrcoef(Y_val, Y_pre)
            except:
                corr = 0
            r_2 = corr ** 2
            if verbose: print("R^2 = %s" % (r_2))

        if verbose: print("Finish training")

        return r_2


class RFEstimator(SKLBGCEstimator):

    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=2):

        super(RFEstimator, self).__init__()
        self.kernel = RandomForestRegressor(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth
        )


class AdaBoostEstimator(SKLBGCEstimator):

    def __init__(self,
                 n_estimators=10,
                 loss='linear'):

        super(AdaBoostEstimator, self).__init__()
        self.kernel = AdaBoostRegressor(
                    n_estimators=n_estimators,
                    loss=loss
        )


class PhyBasBgcEstimator(object):
    """
        基于体质分类血糖估计器
    """

    def __init__(self, bgc_estimator, pws_classifier):
        self.bgc_estimator_init = bgc_estimator
        self.pws_classifier = pws_classifier
        self.labels = []
        self.bgc_estimators = {}

    def fit(self, X_train, Y_train, S_train, X_val=None, Y_val=None, S_val=None, verbose=True):
        S_train_label = self.pws_classifier.predict(S_train)

        self.labels = sorted(list(set(S_train_label)))
        for la in self.labels:

            if verbose: print("Train the %s type bgc_estimator." % (la))
            self.bgc_estimators[la] = copy.deepcopy(self.bgc_estimator_init)
            X_train_tmp, Y_train_tmp = X_train[S_train_label == la], Y_train[S_train_label == la]
            if X_val is None or Y_val is None or S_val is None:
                X_val_tmp, Y_val_tmp = None, None
            else:
                S_val_label = self.pws_classifier.predict(S_val)
                X_val_tmp, Y_val_tmp = X_val[S_val_label == la], Y_val[S_val_label == la]
            self.bgc_estimators[la].fit(
                X_train=X_train_tmp,
                Y_train=Y_train_tmp,
                X_val=X_val_tmp,
                Y_val=Y_val_tmp,
                verbose=verbose
            )

        if verbose:
            print("Finish training. There %s bgc estimators" % (len(self.labels)))

    def predict(self, X, S):

        S_label = self.pws_classifier.predict(S)
        Y_pred = np.zeros(len(X))
        for la in self.bgc_estimators:
            X_tmp = X[S_label == la]
            Y_pred_tmp = self.bgc_estimators[la].predict(X_tmp)
            Y_pred[S_label == la] = Y_pred_tmp

        return Y_pred


if __name__ == "__main__":
    knn_pws_cls = KnnPwsClassification(n_neighbors=10)
    knn_pws_cls.fit(features_file='./features_pws_types.csv')
    type_predict = knn_pws_cls.predict(
        [[0.0543290783838478, 0.3803394046605291, 0.926099522487631, 0.6731195489158396, 0.2771035161263539]])
    print(type_predict)
