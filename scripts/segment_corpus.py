#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: jidognzhang <jidongzhang@tencent.com>
#          andybliu <andybliu@tencent.com>
# Create: 2018/07/27
#
import argparse
import logging
import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SMARTCAT_DIR = os.path.dirname(CURR_DIR)
sys.path.append(SMARTCAT_DIR)

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
    parser.add_argument('--raw_corpus', type=str, required=True,
                        help='raw corpus path')
    parser.add_argument('--seg_corpus', type=str, required=True,
                        help='segmented corpus path')
    parser.add_argument('--title_col', type=int, default=-1,
                        help='column index of title in raw corpus')
    parser.add_argument('--cont_col', type=int, default=-1,
                        help='column index of content in raw corpus')
    return parser.parse_args()


def segment_file(tp, raw_corpus, seg_corpus, title_col=-1, cont_col=-1):
    max_col = max([title_col, cont_col])
    fout = open(seg_corpus, 'w')
    with open(raw_corpus) as fin:
        for i, line in enumerate(fin):
            if i % 10000 == 0:
                logging.info('Finished loading of %d lines' % i)

            fields = line.strip('\n').decode('utf8').split('\t')
            assert (max_col < len(fields))

            out_fields = []
            for i in range(len(fields)):
                if i == title_col:
                    title = fields[i].encode('utf8')
                    seg_title = ' '.join(tp.seg_text(title)[0])
                    out_fields.append(seg_title)
                elif i == cont_col:
                    cont = fields[i]
                    cont = tp.replace_punc(cont, '。')
                    cont = tp.filter_affix(cont)
                    cont = tp.truncate(cont)
                    cont = cont.encode('utf8')
                    seg_cont = ' '.join(tp.seg_text(cont)[0])
                    out_fields.append(seg_cont)
                else:
                    out_fields.append(fields[i])

            fout.write('%s\n' % '\t'.join(out_fields))
    fout.close()
             

if __name__ == '__main__':
    args = process_args()

    tp = TextProcessor('/data/ainlp/classification/data/')

    logging.info('Start segmenting corpus ...')
    logging.info('raw_corpus: %s' % args.raw_corpus)
    logging.info('seg_corpus: %s' % args.seg_corpus)

    segment_file(tp,
				 args.raw_corpus,
				 args.seg_corpus,
				 args.title_col,
				 args.cont_col)

    logging.info('Finished')

