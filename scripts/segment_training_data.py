#!/usr/bin/env python
# Authors: 
# Create: 2018/07/27
#
import argparse
import logging
import sys

sys.path.append('../')
from text_processing import TextProcessor

reload(sys)
sys.setdefaultencoding('utf8')

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True,
                        help='raw training corpus path')
    parser.add_argument('--seg_train_file', type=str, required=True,
                        help='segmented training data path')
    return parser.parse_args()

def seg_file(tp, train_file, seg_train_file):
    fw = open(seg_train_file, 'w')
    with open(train_file) as f:
        for line in f:
            cmsid, label, title, content = line.strip('\n').split('\t')
            fw.write(cmsid + '\t' + label + '\t' + ' '.join(tp.seg_text(title)[0])
                     + '\t' + ' '.join(tp.seg_text(content)[0]) + '\n')
    fw.close()
             

if __name__ == '__main__':
    args = process_args()
    print "train_file:"+args.train_file
    print "train_file_seg:"+args.seg_train_file
    tp = TextProcessor("/data/ainlp/classification/data/new_first_data/")
    seg_file(tp, args.train_file, args.seg_train_file)

    # TODO: segment raw training file and save result.

