# coding:utf-8
from collections import defaultdict

def recall_analysis():
    ft_error_count = defaultdict(int)
    svm_error_count = defaultdict(int)
    pa_error_count = defaultdict(int)
    nb_error_count = defaultdict(int)
    with open(label_file,'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cmsid = parts[0]
            gloden_label = parts[1]
            nb_pred = parts[2]
            pa_pred = parts[3]
            svm_pred = parts[4]
            ft_pred = parts[5]
            if ft_pred == target_label and gloden_label != target_label:
                ft_error_count[gloden_label] += 1
            if svm_pred == target_label and gloden_label != target_label:
                svm_error_count[gloden_label] += 1
            if pa_pred == target_label and gloden_label != target_label:
                pa_error_count[gloden_label] += 1
            if nb_pred == target_label and gloden_label != target_label:
                nb_error_count[gloden_label] += 1
    print("ft",ft_error_count)
    print("nb",nb_error_count)
    print("pa",pa_error_count)
    print("svm",svm_error_count)


def precision_analysis():
    error_count = defaultdict(int)
    with open(label_file,'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            cmsid = parts[0]
            gloden_label = parts[1]
            nb_pred = parts[2]
            pa_pred = parts[3]
            svm_pred = parts[4]
            ft_pred = parts[5]
            if svm_pred != target_label and gloden_label == target_label:
                error_count[svm_pred] += 1
    error_count_items = sorted(error_count.items(), key=lambda x:x[1], reverse=True)
    for label, count in error_count_items:
        print(label + ':' + str(count))


if __name__ == '__main__':
    label_file = 'v2.1.label'
    target_label = 'ent'
    recall_analysis()
    precision_analysis()
