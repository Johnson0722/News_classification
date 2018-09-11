# coding:utf-8

import argparse
import logging
import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from feature_processing import FeatureProcessor

reload(sys)
sys.setdefaultencoding('utf8')

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

VALID_LABELS = [
    'abroad', 'agriculture', 'astro', 'auto', 'baby', 'beauty',
    'career', 'comic', 'creativity', 'cul', 'digital', 'edu',
    'emotion', 'ent', 'finance', 'food', 'funny', 'game', 'health',
    'history', 'house', 'houseliving', 'inspiration', 'law', 'life',
    'lifestyle', 'lottery', 'mil', 'pet', 'photography', 'politics',
    'religion', 'science', 'social', 'sports', 'tech', 'travel',
    'weather', 'women'
]

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        help='training corpus location')
    parser.add_argument('--feature_dict', type=str, required=True,
                        help='feature dict output path')
    parser.add_argument('--rebuild_feature', action='store_true',
                        default=False, help='feature dict output path')
    parser.add_argument('--new_feature_dict', type=str,
                        help='new feature dict output path')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='min value of ngram range')
    parser.add_argument('--ngram_max', type=int, default=1,
                        help='max value of ngram range')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='max number of ngram features')
    return parser.parse_args()

def load_corpus(fname):
    docs = []
    labels = []
    with open(fname) as fin:
        for i, line in enumerate(fin):
            if (i + 1) % 10000 == 0:
                logging.info('Finished loading of %d lines' % i)
            fields = line.decode('utf8').strip('\n').split('\t')
            if len(fields) != 4:
                continue
            cmsid, label, title, content = fields
            text = ' '.join([title, content]).strip()
            label = label.replace('__label__', '')
            if label not in VALID_LABELS:
                continue
            docs.append(text)
            labels.append(label)
    logging.info('Finished loading of corpus, %d lines in total'
                 % len(docs))
    return docs, labels

def corpus_generator(fname):
    with open(fname) as fin:
        for i, line in enumerate(fin):
            fields = line.strip('\n').decode('utf8').split('\t')
            if len(fields) != 4:
                continue
            cmsid, label, title, content = fields
            text = ' '.join([title, content]).strip()
            label = label.replace('__label__', '')
            if label not in VALID_LABELS:
                continue
            yield text

if __name__ == '__main__':
    args = process_args()
    # init feature processor object
    feature_data_dir = '../data/feature_data'
    # rebuild feature mode
    if args.rebuild_feature:
        if args.feature_dict is None:
            raise ValueError('In rebuilding feature mode, feature_dict is needed')
        feature_processor = FeatureProcessor(feature_data_dir, args.feature_dict,
                                             rebuild_feature = True)
        # Dump feature dict
        logging.info('Dumping new feature dict(%s)...' % args.new_feature_dict)
        if args.new_feature_dict:
            feature_processor.dump_features(args.new_feature_dict)
    elif args.train_file is not None:
        ngram_range = (args.ngram_min, args.ngram_max)
        feature_processor = FeatureProcessor(feature_data_dir,
                                             ngram_range=ngram_range)
        # Fit n-gram vectorizer
        logging.info('Fitting corpus...')
        feature_processor.fit(docs=corpus_generator(args.train_file),
                              min_df=10,
                              max_features=args.max_features)
        # Do feature selection and create feature dict
        logging.info('Doing feature selection...')
        feature_processor.feature_selection()
        # Dump feature dict.
        logging.info('Dumping feature dict(%s) ...' % args.feature_dict)
        if args.feature_dict:
            feature_processor.dump_features(args.feature_dict)
    else:
        raise ValueError('Wrong arguments, reaching non-rebuilding_feature'
                         ' mode nor training mode!')
    logging.info('Finished!')