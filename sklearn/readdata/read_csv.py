
import pandas as pd
from scipy.io import arff
from mypython.compute_ranged_acc_and_plot import *

def read_data(path=None, row_index=None, data_index=None, label_index=None, header=None, sep=','):
    if path.split('.')[(-1)] == 'arff':
        data, meta = arff.loadarff(path)
        datas = pd.DataFrame(data)
    else:
        f = open(path, encoding='utf-8')
        datas = pd.read_csv(f, header=header, sep=sep)
    if row_index is not None:
        datas = datas.iloc[row_index]
        if isinstance(row_index, str) or isinstance(row_index[0], str):
            datas = datas.loc[row_index]
    elif label_index is not None:
        data = datas.iloc[:, :-1]
        label = datas.iloc[:, -1]
    else:
        data = datas.iloc[:, :]
        label = np.zeros(len(datas.iloc[:, -1]))
    if not data_index or isinstance(data_index, int) or isinstance(data_index[0], int):
        data = datas.iloc[:, data_index]
    else:
        pass
    if isinstance(data_index, str) or isinstance(data_index[0], str):
        data = datas.loc[:, data_index]
    if not label_index or isinstance(label_index, int) or isinstance(label_index[0], int):
        label = datas.iloc[:, label_index]
    else:
        if isinstance(label_index, str) or isinstance(label_index[0], str):
            label = datas.loc[:, label_index]
        return (data, label)


if __name__ == '__main__':
    path = 'D:\\ChromeDownloads\\forestfires.csv'
    row_index = None
    data_index = None
    data_index = None
    label_index = None
    read_data(path=path, row_index=row_index, data_index=data_index, label_index=label_index)
# okay decompiling read_csv.pyc
