# uncompyle6 version 3.7.4
# Python bytecode 3.8 (3413)
# Decompiled from: Python 3.8.5 (default, Sep  3 2020, 21:29:08) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\Users\px\Desktop\zuoye\scikit-learn\sklearn\mypython\dataprocessing.py
# Compiled at: 2021-01-09 17:22:21
# Size of source mod 2**32: 7230 bytes
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.RandomState(42)


def get_data_label(data_address=None, b_row=None, e_row=None, b_column=None, e_column=None, label_negative_index=None,
                   label_column=None):
    data_name = data_address.split('/')[(-1)][:-4]
    data_type = data_address.split('.')[(-1)]
    if data_type != 'npz':
        data_origin = np.array(pd.read_csv(data_address, header=None).iloc[1:, 1:].replace(to_replace='?',
                                                                                           value=(np.nan)).dropna(
            how='any'))
        data = data_origin[b_row:e_row, b_column:e_column]
        if label_negative_index:
            label = np.ones((len(data_origin)), dtype=int)
            label[label_negative_index] = -1
        else:
            label = data_origin[:, label_column]
    else:
        data_label = np.load(data_address)
        print(data_label.files)
        print(data_label['train_labels'].shape, data_label['test_labels'].shape)
        data = np.vstack((data_label['train_images'], data_label['test_images']))
        label = np.vstack((data_label['train_labels'], data_label['test_labels'])).reshape(-1).astype(int)
        print(label)
        label[label == 1] = -1
        label[label == 0] = 1
        print('\n', label)
    return (data, label)


def lof(contamination=None, test_data=None, train_data=None, n_neighbors=50, novelty=False):
    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)
        print('data.ndim=1')
    else:
        if test_data.ndim == 3:
            test_data = test_data.flatten().reshape(-1, 784)
        elif contamination:
            contamination_lof = contamination
            clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination_lof)
        else:
            clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        if train_data:
            clf.fit(train_data)
            predict = clf.predict(test_data)
            score1 = clf.negative_outlier_factor_
            score2 = clf.decision_function(test_data)
            score = score2
            predict_proba = clf.predict_proba(test_data)
        else:
            predict = clf.fit_predict(test_data)
        score = clf.negative_outlier_factor_
    return (score, predict)




# 按区间计算acc表现
def acc_list_compute(per_split=None, label=None, *label_pre):
    acc_list_j = []
    n_samples = len(label)
    k_split = int(n_samples / per_split)
    #print("label_pre shape:{}".format( np.shape(label_pre)))
    #label_pre = np.array(label_pre).ravel()
    for j in label_pre:
        acc_list_i = []
        j = np.array(j).ravel()
        for i in np.arange(k_split):
            begin = i * per_split
            end = (i + 1) * per_split
            acc_list_i.append(acc_per_compute(label[begin:end], j[begin:end]))
        if end != len(label):
            begin = end
            acc_list_i.append(acc_per_compute(label[begin:], j[begin:]))
        acc_list_j.append(acc_list_i)
    #print(np.array(acc_list_j))
    #print(np.shape(acc_list_j))
    return acc_list_j


def acc_per_compute(label=None, *label_pre, degree=6):
    label = np.array(label)
    acc = []
    for i in label_pre:
        #for j in i:
        label_pre_j = np.array(i).ravel()
        errors = (label != label_pre_j).sum()
        #print(round((1 - errors / len(label)) * 100, degree))
        acc.append(round((1 - errors / len(label)) * 100, degree))

    # else:
    #     label_pre = np.array(label_pre)
    #     errors = (label != label_pre).sum()
    #     #print(round((1 - errors / len(label)) * 100, degree))
    #     acc = round((1 - errors / len(label)) * 100, degree)

    return acc


# 按行显示acc结果
def acclist_show(acc_list, n=4):
    print(type(acc_list),len(acc_list))
    for acc_list_i in acc_list:
        # acc_list_i = np.array(acc_list_i).ravel()
        for i in range(n):
            for j in range(int(len(acc_list_i) / n)+1):
                print((acc_list_i[i * int(len(acc_list_i) / n) + j]), end=',')
            print('')

   # else:
   #     for i in range(n):
   #         for j in range(int((acc_list)/ n)):
   #             print((acc_list[4 * i + j]), end=',')
   #         else:
   #             print('')
#True 降序方式排列，即以偏正常到偏异常排列
def get_range_index(score=None, reverse =True):
    index = []
    c = 0
    for i in sorted(score, reverse=reverse):
        if np.where(score == i)[0][0] in index:
            continue
        else:
            index = np.append(index, np.where(score == i)[0]).astype(dtype=int)
    else:
        return index


def range_data(data=None, index=None):
    data = np.array(data)
    #print(data,index)
    data_ranged = data[index]
    return data_ranged


# 对lael,label_pre按index进行排序,返回排序完的label
def range_list(label=None, *label_pre, index=None):
    list = []
    #print(np.array(label_pre).ndim)
    #if np.ndim(label_pre) == 3:
    for i in label_pre:
        i = np.array(i).ravel()
        i = np.array(i)
        list.append(range_data(i, index=index))
    # else:
    #     label_pre = np.array(label_pre).ravel()
    #     list.append(range_data(label_pre, index=index))
    label_ranged = range_data(label, index=index)
    return label_ranged, list


def label_pre_combine(label_pre_if, label_pre_lof, propotion=0.5):
    le = int(len(label_pre_if) * propotion)
    predict_combine = np.append(label_pre_if[:le], label_pre_lof[le:])
    return predict_combine


def del_data(*data, index=None, contamination=0.5):
    k = ()
    for i in data:
        i = np.array(i)
        le = int(len(i) * contamination)
        del_index = index[:le]
        if i.ndim > 1:
            i_del = i[del_index, :]
            i_new = np.delete(i, del_index, axis=0)
        else:
            i_del = i[del_index]
            i_new = np.delete(i, del_index)
        k += (i_del, i_new)
    else:
        return k

#传入标签数组和类别，获得对应类的索引，以字典形式进行返回
def get_label_index(label,*category):
    index_dic = {}
    if np.size(category) !=1:
        for i in category:
            for j in i:
                index_dic[j]=list(np.where(label == j)[0])
    else:
        for j in category:
            index_dic[j] = list(np.where(label == j)[0])
    return index_dic
#将label字典传入,设置每一类用于训练的比例列表,返回训练类和对应索引
def get_train_index(label,propotion_list=None):
    if isinstance(label,dict):
        dic = {}
        j = 0
        for i in label:
            if propotion_list is not None:
                dic[i] = train_test_split(label[i],train_size=0.8)[0]
                j+=1
            else:
                dic[i] = train_test_split(label[i],train_size=0.8)[0]
        return dic
    else:
        if propotion_list:
            return train_test_split(label,train_size=0.8)
        else:
            return train_test_split(label)
#传入带有类和对应索引的dict,计算每一类的acc并以字典形式返回
def compute_category_acc(dict_label_index,label,pre_label,train_label_index):
    dic={}
    j=[]
    for i in dict_label_index:
        dic[i]=acc_per_compute(pre_label[dict_label_index[i]],label[dict_label_index[i]])
    dic['train_data']=acc_per_compute(pre_label[train_label_index],label[train_label_index])
    dic['acc']=acc_per_compute(label,pre_label)
    return dic

def label_process(label,category):
    for i in category:
        label[label==i] = 1
    label[label!=1] = -1
    return label

#获得训练类的标签索引
def get_label_index_list(dic):
    lst = []
    for i in dic:
        lst+=dic[i]
    return lst

#传入多个score数组或者acc，排序并合并返回排序后的score_list，acc_list
def pack_list(*score,reverse = False):
    lis = []
    for i in score:
        if reverse:
            a = sorted(i,reverse=reverse)
        else:
            a = i
        lis.append(a)
    return lis

# 传入acc和各种未排序异常得分，越偏正常位置越靠近0，按acc/index进行比较，值越大的，设置为标签，返回阿尔法系数列表。
def compute_a_list(score_list, acc_list=None ,flag = 1):
    a_list = []
    for k in range(len(score_list)):
        dic = {}
        a = []
        # 对score进行排序，越正常越靠近左边
        index = get_range_index(score_list[k])
        #index = score_list[k]
        for i in range(len(index)):
            dic[index[i]] = i+1
        for j in range(len(index)):
        #计算原样本对应算法的阿尔法系数，由1/index定义,即越右边越正常
            #print(acc_list[k],dic[j])
            if acc_list is not None and flag == 1:
                a.append(acc_list[k][0]/dic[j])
            else:
                a.append(1/dic[j])
        a_list.append(a)
    return a_list
#返回各个算法对应于原样本未排序的阿尔法系数列表,得分排名越靠左边越可能是正常数据

#选择阿尔法系数较大者作为predict标签,返回预测标签

def compute_vote_pre(pre_list):
    pre_list = pre_list+1
    count1 = np.count_nonzero(pre_list,axis=0)  #统计每一列1的个数
    vote_pre = np.where(count1 >= len(pre_list)-count1, 1, -1).tolist()
    return vote_pre

#传入阿尔法系数列表，及对应的预测标签,
#返回w0(不带权),w1(带权),w2(vote产生)的预测标签pre_list

def compute_pre(a_list,pre_list,weight=0):
# lis.append(score_list[i]/acc_list[i])
    pre_w0 = []
    pre_w1 = []
    if weight == 0:
        a_max_list = np.argmax(a_list,axis=0)    #计算最大阿尔法系数对应的算法
        for i in range(len(a_max_list)):
            pre_w0.append(pre_list[a_max_list[i]][i])
        #vote_pre = compute_vote_pre(pre_list)


        lis = a_list * pre_list #计算系数和标签乘积，对每一列求和，>=0则标签置为1，否则-1
        lis = np.sum(lis,axis=0)
        pre_w1 = np.where(lis>=0,1,-1)
        vote_pre = compute_vote_pre(pre_list)
        return pre_w0,pre_w1,vote_pre

#按index位置的方法，返回对应index_list对应的值
def f(index,args):
    return np.array(args)[index]

#获得index索引下的对应list
def get_list(index,*args):
    if isinstance(args,tuple):
        a=list(map(lambda i: f(index,i),args))
        return a
    return f(index,args)

#传入得分结果，用异常检测方法对得分结果进行二次检测，并将检测结果返回成score_probe
def probe(score_list,clf,clf_name = 'lof'):
    score_probe = []
    for i in score_list:
        i = i.reshape(-1,1)
        clf.fit(i)
        score_probe.append(clf.decision_function(i))
    return score_probe