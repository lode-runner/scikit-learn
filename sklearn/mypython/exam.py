# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.0 (default, Nov  6 2019, 16:00:02) [MSC v.1916 64 bit (AMD64)]
#
# Compiled at: 2021-04-05 17:26:50
# Size of source mod 2**32: 8666 bytes
from scipy.io import loadmat
from mypython.dataprocessing import *
from readdata.read_csv import *
from sklearn.ensemble import AdaBoostClassifier
from pyod.models.knn import KNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn import svm
from pyod.models.hbos import HBOS
from pyod.models.auto_encoder import AutoEncoder
import tensorflow
class A:

    def __init__(self, score_list=None, acc_train_list=None, acc_test_list=None, acc_list=None, name_list=None, pre_train_list=None, pre_list=None):
        self.score_list = score_list, self.acc_train_list = acc_train_list, self.acc_test_list = acc_test_list, self.acc_list = acc_list, self.name_list = name_list, self.pre_train_list = pre_train_list, self.pre_list = pre_list

    np.random.seed(40)
    flag = -1
    while flag != 0:
        flag += 1
        data_name_list = ('arrhythmia.matarrhythmia', 'cardio.matcardio', 'ionosphere.mationosphere',
                  'lympho.matlympho', 'satimage-2.matsatimage-2','speech.matspeech',
                   'wbc.matwbc','satellite.matsatellite','shuttle.matshuttle')

        for k in range(len(data_name_list)):
                kk = data_name_list[k]
                method_list = [[0,1,2],[0,1,3],[0,1,2],[1,2,3],[0,2,3],
                               [1,2,3],[1,2,3],[0,1,3],[0,1,3]]
                method = method_list[k]
                path = 'C:\\Users\\user\\Desktop\\data\\'+kk+'.mat'
                data_name = path.split('.')[-2:]
                np.random.seed(40)
                row_index = None
                data_index = None
                data_index = None
                label_index = None
                f = loadmat(path)
                data = f['X']
                label = f['y']
                data = pd.DataFrame(data)
                label = label.tolist()
                label = np.array(label).reshape(-1)
                label[label == 1] = -1
                label[label == 0] = 1
                x = data.copy()
                y = label.copy()

                label_dic = get_label_index(label, [1, -1])
                label_dic1 = label_dic[1]
                label_dic_1 = label_dic[(-1)]
                x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.6)
                i, j = int(0.6 * len(label_dic1)), int(0.6 * len(label_dic_1))
                label1 = label_dic1[:i]
                label2 = label_dic_1[:j]
                contamination = j / (i + j)
                #contamination = np.sum([y_train == -1]) / len(y)
                name_list = []
                score_list = []
                pre_train_list = []
                pre_list = []
                acc_train_list = []
                acc_test_list = []
                acc_list = []

                i = 0
                clf_name = 'if'
                name_list.append(clf_name)
                clf = IsolationForest()
                clf.fit(x_train)
                score = clf.decision_function(x)
                pre_train = clf.predict(x_train)
                pre_test = clf.predict(x_test)
                pre = clf.predict(x)
                acc_train = acc_per_compute(y_train, pre_train)
                acc_test = acc_per_compute(y_test, pre_test)
                acc = acc_per_compute(y, pre)
                score_list.append(score)
                pre_train_list.append(pre_train)
                pre_list.append(pre)
                acc_train_list.append(acc_train)
                acc_test_list.append(acc_test)
                acc_list.append(acc)


                i = 1
                clf_name = 'AutoEncoder'
                n = int(np.shape(x_train)[1] / 2)
                hidden_neurons = None
                if n < 32:
                    hidden_neurons = [n, 2 * n, 2 * n, n]
                clf = AutoEncoder(hidden_neurons=hidden_neurons)
                name_list.append(clf_name)
                clf.fit(x_train)
                score = clf.decision_function(x)
                pre_train = clf.predict(x_train)
                pre_test = clf.predict(x_test)
                pre = clf.predict(x)
                pre_train[pre_train == 1] = -1
                pre_train[pre_train == 0] = 1
                pre_test[pre_test == 1] = -1
                pre_test[pre_test == 0] = 1
                pre[pre == 1] = -1
                pre[pre == 0] = 1
                acc_train = acc_per_compute(y_train, pre_train)
                acc_test = acc_per_compute(y_test, pre_test)
                acc = acc_per_compute(y, pre)
                score_list.append(score)
                pre_train_list.append(pre_train)
                pre_list.append(pre)
                acc_train_list.append(acc_train)
                acc_test_list.append(acc_test)
                acc_list.append(acc)

                i = 2
                clf_name = 'KNN'
                clf = KNN()
                name_list.append(clf_name)
                clf.fit(x_train)
                score = clf.decision_function(x)
                pre_train = clf.predict(x_train)
                pre_test = clf.predict(x_test)
                pre = clf.predict(x)
                pre_train[pre_train == 1] = -1
                pre_train[pre_train == 0] = 1
                pre_test[pre_test == 1] = -1
                pre_test[pre_test == 0] = 1
                pre[pre == 1] = -1
                pre[pre == 0] = 1
                acc_train = acc_per_compute(y_train, pre_train)
                acc_test = acc_per_compute(y_test, pre_test)
                acc = acc_per_compute(y, pre)
                score_list.append(score)
                pre_train_list.append(pre_train)
                pre_list.append(pre)
                acc_train_list.append(acc_train)
                acc_test_list.append(acc_test)
                acc_list.append(acc)

                i=3
                clf_name = 'HBOS'
                name_list.append(clf_name)
                clf = HBOS()
                clf.fit(x_train)
                score = clf.decision_function(x)
                pre_train = clf.predict(x_train)
                pre_test = clf.predict(x_test)
                pre = clf.predict(x)
                pre_train[pre_train == 1] = -1
                pre_train[pre_train == 0] = 1
                pre_test[pre_test == 1] = -1
                pre_test[pre_test == 0] = 1
                pre[pre == 1] = -1
                pre[pre == 0] = 1
                acc_train = acc_per_compute(y_train, pre_train)
                acc_test = acc_per_compute(y_test, pre_test)
                acc = acc_per_compute(y, pre)
                score_list.append(score)
                pre_train_list.append(pre_train)
                pre_list.append(pre)
                acc_train_list.append(acc_train)
                acc_test_list.append(acc_test)
                acc_list.append(acc)
                # i = 4
                # clf_name = 'AdaBoost'
                # name_list.append(clf_name)
                # clf = AdaBoostClassifier()
                # clf.fit(x_train, y_train)
                # score = clf.decision_function(x)
                # pre_train = clf.predict(x_train)
                # pre_test = clf.predict(x_test)
                # pre = clf.predict(x)
                # acc_train = acc_per_compute(y_train, pre_train)
                # acc_test = acc_per_compute(y_test, pre_test)
                # acc = acc_per_compute(y, pre)
                # score_list.append(score)
                # pre_train_list.append(pre_train)
                # pre_list.append(pre)
                # acc_train_list.append(acc_train)
                # acc_test_list.append(acc_test)
                # acc_list.append(acc)
                #
                # i = 5
                # clf_name = 'svm'
                # name_list.append(clf_name)
                # clf = svm.SVC(kernel='linear')
                # clf.fit(x_train, y_train)
                # score = clf.decision_function(x)
                # pre_train = clf.predict(x_train)
                # pre_test = clf.predict(x_test)
                # pre = clf.predict(x)
                # acc_train = acc_per_compute(y_train, pre_train)
                # acc_test = acc_per_compute(y_test, pre_test)
                # acc = acc_per_compute(y, pre)
                # score_list.append(score)
                # pre_train_list.append(pre_train)
                # pre_list.append(pre)
                # acc_train_list.append(acc_train)
                # acc_test_list.append(acc_test)
                # acc_list.append(acc)

                score_list, acc_train_list, acc_test_list, acc_list, name_list, pre_train_list, pre_list =\
                    get_list(method, score_list, acc_train_list, acc_test_list, acc_list, name_list, pre_train_list, pre_list)
                #print(name_list)


                clf = IsolationForest()
                score_probe = probe(score_list, clf)

                print('data_name:%s.%s\t%s'%(data_name[0],data_name[1],k))
                a_list = compute_a_list(score_probe, acc_train_list, flag=flag)
                # 计算w0(不带权),w1(带权),w2(vote产生)的预测标签pre_list
                pre = compute_pre(a_list, pre_list)
                #计算w0,w1,w2 acc并输出
                acc_list_c_show(label, *pre, label_name=name_list)


                #输出各算法acc
                for i in range(len(name_list)):
                    print('acc_%s:%s,train:%s,test:%s'%(name_list[i],acc_list[i],acc_train_list[i],acc_test_list[i]))

        def f(self, score_list=score_list, acc_train_list=acc_train_list, acc_test_list=acc_train_list, acc_list=acc_list, name_list=acc_list, pre_train_list=acc_train_list, pre_list=pre_list):
                    self.score_list = score_list, self.acc_train_list = acc_train_list, self.acc_test_list = acc_test_list, self.acc_list = acc_list, self.name_list = name_list, self.pre_train_list = pre_train_list, self.pre_list1 = pre_list

