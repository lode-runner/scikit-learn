# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\Users\px\Desktop\zuoye\scikit-learn\sklearn\mypython\dataplot.py
# Compiled at: 2021-01-06 09:09:59
# Size of source mod 2**32: 831 bytes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plt_show(x, y=None, label=None, i=0, name=None):
    matplotlib.rcParams['font.family']='SimHei'
    if i == 0:
        plt.plot(x, label=label)
        plt.plot(np.zeros(len(x))+50, label='acc=50%')
        plt.plot(np.zeros(len(x)) + 75, label='acc=75%')
        legend = plt.legend(loc=0)
        plt.ylim(-1, 105)
    else:
        print(np.shape(x))
        y = np.arange(len(x))
        radius = (x.max() - x) / (x.max() - x.min())
        x = (x - np.mean(x)) / np.std(x)
        plt.scatter(y, x, label=label)
        plt.scatter(y, x, s=(1000 * radius), edgecolors='r', label='outlierscore', facecolors='none')
        plt.axis('tight')
        legend = plt.legend(loc=0)
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [10]
    plt.show()
#传入字典，将对应的类画在同一图上
def plot_dict_show(dict):
    lenth = 0
    for i in dict:
        plt.plot(dict[i],label = i)
        lenth +=len(dict[i])
    plt.plot(np.zeros(100) + 50, label='acc=50%')
    plt.plot(np.zeros(100) + 75, label='acc=75%')
    legend = plt.legend(loc=0)
    plt.ylim(-1, 105)
    plt.show()
# okay decompiling dataplot.cpython-38.pyc
