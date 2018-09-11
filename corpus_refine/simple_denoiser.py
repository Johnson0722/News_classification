# coding:utf-8
import random

TRAIN_FILE = "local_data/cat_train_v1.3.seg20180713"
OUTPUT_TRAIN_FILE = "local_data/cat_train_v1.4.seg20180713"
OUTPUT_FT_FILE = 'local_data/cat_train_v1.4.seg20180713.ft'

KEY_WORDS = [['空客','旅客'],['旅客','客机'],['春运'],
             ['海南航空'],['东方航空'],['南方航空'],['中国国际航空'],['东方航空'],['河北航空'],['四川航空'],
             ['天津航空'],['海航集团'],['华夏航空'],['广州白云机场'],
             ['飞机','托运'],['飞机','旅游'],['飞机','经济舱']]

def random_dropout(dropout_rate, cate):
    """random dropout samples from given category with dropout rate
    :param dropout_rate: between 0 and 1, type of float
    :param cate: specific category, type of str
    """
    error_lines = 0
    count_lines = 0
    fout = open(OUTPUT_TRAIN_FILE, 'w')
    ft_out = open(OUTPUT_FT_FILE, 'w')
    with open(TRAIN_FILE, 'r') as fin:
        for line in fin:
            try:
                parts = line.strip('\n').split('\t')
                cmsid = parts[0]
                label = parts[1]
                title = parts[2]
                content = parts[3]
                if label == cate:
                    num = random.random()
                    if num < dropout_rate:
                        continue
                    fout.write(line)
                    ft_out.write(label + ' ' + title + ' ' + content + '\n')
            except Exception as e:
                error_lines += 1
                print(e)
            finally:
                count_lines += 1
                if count_lines % 100000 == 0:
                    print('{} lines has finished'.format(count_lines))
                    print('{} error lines'.format(error_lines))

def keywords_dropout(keywords):
    """drop lines if title or content contain any keywords"""
    noise_lines = 0
    count_lines = 0
    fout = open(OUTPUT_TRAIN_FILE, 'w')
    ft_out = open(OUTPUT_FT_FILE, 'w')
    with open(TRAIN_FILE, 'r') as fin:
        for line in fin:
            parts = line.strip('\n').split('\t')
            cmsid = parts[0]
            label = parts[1]
            title = parts[2].split()
            content = parts[3].split()
            words = title + content
            for keyword in keywords:
                if set(keyword) | set(words) == set(words):
                    noise_lines += 1
                else:
                    fout.write(line)
                    ft_out.write(label + ' ' + title + ' ' + content + '\n')
                count_lines += 1
                if count_lines % 100000 == 0:
                    print('{} lines have finished'.format(count_lines))

if __name__ == '__main__':
    pass
