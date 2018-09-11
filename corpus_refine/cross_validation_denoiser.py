# coding:utf-8
# baby&health denoising using fasttext bi-classifier, 10-fold cross validation
import random
import logging
import fasttext

TRAIN_FILE = "local_data/cat_train_v1.9.seg20180713"
OUTPUT_FILE_HELATH = 'tools/health_noise_cmsid.txt'
OUTPUT_FILE_BABY = 'tools/baby_noise_cmsid.txt'
TEMP_TRAIN_FILE = 'tools/temp_ft_train.txt'

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

def loading_bi_classifier_training_data():
    logging.info("start loding data...")
    all_lines = 0
    positive_lines = []
    negative_lines = []
    with open(TRAIN_FILE,'r') as fin:
        for line in fin:
            parts = line.strip('\n').split('\t')
            label = parts[1]
            if label == '__label__health':
                positive_lines.append(line)
            if label == '__label__baby':
                negative_lines.append(line)
            all_lines += 1
            if all_lines % 100000 == 0:
                logging.info('{} has finished'.format(all_lines))
    return positive_lines, negative_lines

def cross_validation(positive_lines, negative_lines):
    # start validation!
    num_positive = len(positive_lines)
    num_negative = len(negative_lines)
    logging.info('There are {} total positive samples'.format(num_positive))
    logging.info('There are {} total negative samples'.format(num_negative))
    positive_batches = num_positive / 10
    negative_batches = num_negative / 10
    count = 1
    for i in range(10):
        # generate temp train data
        temp_f = open(TEMP_TRAIN_FILE, 'w')
        train_positive_lines = positive_lines[:i*positive_batches] + positive_lines[(i+1)*positive_batches:]
        train_negative_lines = negative_lines[:i*negative_batches] + negative_lines[(i+1)*negative_batches:]
        test_positive_lines = positive_lines[i*positive_batches:(i+1)*positive_batches]
        test_negative_lines = negative_lines[i*negative_batches:(i+1)*negative_batches]
        all_lines = train_positive_lines + train_negative_lines
        random.shuffle(all_lines)
        for line in all_lines:
            parts = line.strip('\n').split('\t')
            label = parts[1]
            title = parts[2]
            content = parts[3]
            temp_f.write(label + ' ' + title + ' ' + content + '\n')
        logging.info('{} validation process train ft...'.format(count))
        temp_ft_model = train_fasttext(TEMP_TRAIN_FILE)
        logging.info('{} validation process test ft...'.format(count))
        test_lines = test_positive_lines + test_negative_lines
        test_fasttext(test_lines, temp_ft_model)
        logging.info("{} validation process has finished".format(count))
        count += 1

def train_fasttext(train_file):
    logging.info("start training FT model...")
    temp_ft_path = '../refine/tools/temp_model.ft'
    temp_ft_model = fasttext.supervised(train_file, temp_ft_path,
                                        label_prefix='__label__')
    logging.info('training ft finished!')
    return temp_ft_model

def test_fasttext(test_lines, ft_model):
    fout_health = open(OUTPUT_FILE_HELATH, 'a')
    fout_baby = open(OUTPUT_FILE_BABY, 'a')
    for line in test_lines:
        parts = line.strip('\n').split('\t')
        cmsid = parts[0]
        label = parts[1].replace('__label__', '')
        title = parts[2]
        content = parts[3]
        text = title + ' ' + content
        pred_label_prob = ft_model.predict_proba([text], k=2)
        pred_label, pred_prob = pred_label_prob[0][0]
        pred_label = pred_label.replace('__label__','')
        if label == 'health':
            if pred_label != label and pred_prob > 0.9:
                fout_health.write(cmsid + '\n')
        if label == 'baby':
            if pred_label != label and pred_prob > 0.9:
                fout_baby.write(cmsid + '\n')

if __name__ == '__main__':
    positive_lines, negative_lines = loading_bi_classifier_training_data()
    cross_validation(positive_lines, negative_lines)
