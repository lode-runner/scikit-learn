import numpy as np
import pandas as pd
from mypython.dataprocessing import *
from mypython.dataplot import *
score = None


#返回按index排序完的label和label_pre,返回参数：label_ranged, list
def label_rank(label,*label_pre,index=None):
    return range_list(label,label_pre,index=index)



#返回分成k_split个区间的acc_list和显示，输入score，将原标签按score排序分区，否则按默认分区间排序
def acc_list_c_show(label=None, *label_pre , score = None,k_split=None,reverse = True, label_name = '默认'):
    if score is not None:
        index = get_range_index(score=score, reverse=reverse)     #reverse=true,偏正常向偏异常排列
        label,label_pre = range_list(label, *label_pre, index=index)
 #   acc_list=acc_list_compute(k_split,label,label_pre)
    #print(acc_list)
    #acclist_show(acc_list)
    acc_list = acc_per_compute(label,*label_pre)
    print('acc_%s:\nacc_w0:%s,  acc_w1:%s,  acc_vote:%s'%(label_name,acc_list[0],acc_list[1],acc_list[2]))

    #plt_show(acc_list[0],label=label_name)

    return acc_list



if __name__=='main':
    index = None
    label = None
    label_pre = None
    k_split = None
    label, label_pre = label_rank(label,label_pre,index=index)
    acc_list_compute(k_split,label,label_pre)


