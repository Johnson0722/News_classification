# coding:utf-8

import random
import logging
import fasttext

# use "(POSITIVE_LABEL,others)" bi-classifier to dropout samples in TARGET LABEL samples
POSITIVE_LABEL = 'ent'
TARGET_LABEL = 'funny'
TRAIN_FILE = "local_data/cat_train_v2.1.seg20180713"
OUTPUT_FILE = 'tools/noise_cmsid.txt'
TEMP_TRAIN_FILE = 'tools/temp_ft_train.txt'
TEMP_FT_FILE = 'tools/temp_model.ft'
# probability of select negative lines
THRESHOLD = 0.15

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

def loading_classifier_training_data(positive_label, target_label):
    logging.info('start loading data')
    count_lines = 0
    positive_lines = []
    negative_lines = []
    target_lines = []
    with open(TRAIN_FILE, 'r') as fin:
        for line in fin:
            parts = line.strip('\n').split('\t')
            label = parts[1].replace('__label__','')
            if label == positive_label:
                positive_lines.append(line)
            elif label == target_label:
                target_lines.append(line)
            else:
                num = random.random()
                if num < THRESHOLD:
                    line = line.replace(label, '__label__others')
                    negative_lines.append(line)
            count_lines += 1
            if count_lines % 100000 == 0:
                logging.info('{} lines have finished'.format(count_lines))
    return positive_lines, negative_lines, target_lines

def train_fasttext(train_file):
    logging.info("start training FT model...")
    temp_ft_model = fasttext.supervised(train_file, TEMP_FT_FILE,
                                        label_prefix='__label__')
    logging.info('training ft finished!')
    return temp_ft_model

def test_fasttext(test_lines, ft_model):
    """write noise cmsid to OUTPUT file"""
    fout = open(OUTPUT_FILE, 'a')
    count = 0
    for line in test_lines:
        parts = line.strip('\n').split('\t')
        cmsid = parts[0]
        title = parts[2]
        content = parts[3]
        text = title + ' ' + content
        pred_label_prob = ft_model.predict_proba([text], k=2)
        pred_label, pred_prob = pred_label_prob[0][0]
        pred_label = pred_label.replace('__label__','')
        if pred_label == 'ent' and pred_prob > 0.99:
            fout.write(cmsid + '\n')
            count += 1
            if count % 2000 == 0:
                logging.info('There are {} ent samples in target corpus'.format(count))

def main(positive_lines, negative_lines, target_lines):
    num_positive = len(positive_lines)
    num_negative = len(negative_lines)
    num_targets = len(target_lines)
    logging.info('There are {} total positive samples'.format(num_positive))
    logging.info('There are {} total negative samples'.format(num_negative))
    logging.info('There are {} total target samples'.format(num_targets))
    # generate temp train data
    temp_f = open(TEMP_TRAIN_FILE, 'w')
    all_lines = positive_lines + negative_lines
    random.shuffle(all_lines)
    for line in all_lines:
        parts = line.strip('\n').split('\t')
        label = parts[1]
        title = parts[2]
        content = parts[3]
        temp_f.write(label + ' ' + title + ' ' + content + '\n')
    logging.info('training fasttext...')
    temp_ft_model = train_fasttext(TEMP_TRAIN_FILE)
    logging.info('testing fasttext')
    test_fasttext(target_lines, temp_ft_model)

if __name__ == '__main__':
    positive_lines, negative_lines, target_lines = \
        loading_classifier_training_data(POSITIVE_LABEL, TARGET_LABEL)
    main(positive_lines, negative_lines, target_lines)
