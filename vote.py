import os
import numpy as np
from sklearn.metrics import confusion_matrix
from stats import *


def load_data(path):
    res = {}
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
    for line in lines:
        id_log_target = line.replace(',', '').replace('[', '').replace(']', '').split()
        idx = id_log_target[0]
        logits = list(map(float, id_log_target[1:4]))
        labels = id_log_target[4]
        preds = logits.index(max(logits))
        if idx in res:
            res[idx]['preds'].append(preds)
        else:
            res[idx] = {
                'logits': logits,
                'preds': [int(preds)],
                'labels': int(labels),
            }
    #     print(idx, logits, preds, labels)
    # print(res)
    return res
        

def cal_CM(res):
    pred_label = np.array([[res[x]['preds'][0], res[x]['labels']] for x in res])
    preds = pred_label[:, 0]
    labels = pred_label[:, 1]
    cm = confusion_matrix(labels, preds)
    # print(cm)
    draw_cm(labels, preds)
    draw_score(labels, preds)
    drawdis(labels, preds)
    get_acc(labels, preds)
        

if __name__ == '__main__':
    res = load_data('test_results/pd_finetune/0.txt')
    cal_CM(res)