#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: andybliu <andybliu@tencent.com>
# Create: 2018/07/27
#

import argparse
import logging
import os
import sys

import fasttext
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB  
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from feature_processing import FeatureProcessor

from smartcat.scripts.train_ensemble.train_xgboost import Ensemble

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True,
                        help='segmented training data path')
    parser.add_argument('--ft_train_file', type=str,
                        help='segmented training data path for fasttext')
    parser.add_argument('--test_file', type=str,
                        help='segmented test data path')
    parser.add_argument('--feature_dict', type=str, required=True,
                        help='feature dict path')
    parser.add_argument('--ngram_max', type=int, default=1,
                        help='max value of ngram range')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='model out dir')
    parser.add_argument('--model_signature', type=str,
                        help='distinguishable signature of model name')
    parser.add_argument('--save_transform', type=str,
                        help='save transfrom to local')
    parser.add_argument('--load_transform', type=str,
                        help='transfrom file path')
    parser.add_argument('--train_xgboost', type=str,
                        help='train xgboost [0,1]')
    parser.add_argument('--feature_mode', type=str,
                        help='xgboost feature')
    return parser.parse_args()


def load_corpus(fname, cols={}):
    label_col = cols.get('label', -1)
    title_col = cols.get('title', -1)
    cont_col = cols.get('content', -1)
    max_col = max([label_col, title_col, cont_col])

    titles = []
    conts = []
    labels = []
    with open(fname) as fin:
        for i, line in enumerate(fin):
            if i % 10000 == 0:
                logging.info('Finished loading of %d lines' % i)

            fields = line.strip('\n').decode('utf8').split('\t')
            if label_col >= 0 and label_col < len(fields):
                labels.append(fields[label_col])
            else:
                labels.append(None)

            if title_col >= 0 and title_col < len(fields):
                titles.append(fields[title_col])
            else:
                titles.append('')

            if cont_col >= 0 and cont_col < len(fields):
                conts.append(fields[cont_col])
            else:
                conts.append('')
    return titles, conts, labels


def train_models(train_X, train_y, test_X=None, test_y=None,
                 model_prefix='./model/temp'):
    # Train {{{.
    logging.info('Training NB model ...')
    nb_model = MultinomialNB(alpha=0.01)
    nb_model.fit(train_X, train_y)

    logging.info('Training SVM model ...')
    svm_model = LinearSVC(random_state=1)
    svm_model.fit(train_X, train_y)

    logging.info('Training PA model ...')
    pa_model = PassiveAggressiveClassifier()
    pa_model.fit(train_X, train_y)
    # }}}.

    # Test {{{.
    if test_X is not None and test_y is not None:
        logging.info('Evaluating on test set ...')
        test_y = [l.replace('__label__', '') for l in test_y]
        for model, desp in zip([nb_model, pa_model, svm_model],
                               ['NB_Report', 'PA_Report', 'SVM_report']):
            print >>sys.stderr, (
                '================== %s ==================\n' % desp)
            pred_y = model.predict(test_X)
            pred_y = [l.replace('__label__', '') for l in pred_y]
            print >>sys.stderr, classification_report(test_y, pred_y, digits=4)
    # }}}.

    # Save models {{{.
    joblib.dump(nb_model, model_prefix + '.nb', compress=True)
    joblib.dump(svm_model, model_prefix + '.svm', compress=True)
    joblib.dump(pa_model, model_prefix + '.pa', compress=True)
    # }}}.


def train_ft(train_file, test_docs, test_y, model_prefix='./model/temp'):
    # Train {{{.
    logging.info('Training FT model ...')

    ft_model_prefix = model_prefix + '.ft'
    ft_model = fasttext.supervised(train_file,
                                   ft_model_prefix,
                                   word_ngrams=2, 
                                   label_prefix='__label__',
                                   bucket=2000000)
    # }}}.

    # Test {{{.
    if test_docs:
        logging.info('Testing FT model ...')
        test_y = [y.replace('__label__', '') for y in test_y]
        pred = ft_model.predict(test_docs)
        pred_y = [item[0] for item in pred]

        print >>sys.stderr, '================= FT_report ==================\n'
        print >>sys.stderr, classification_report(test_y, pred_y, digits=4)
    # }}}.


if __name__ == '__main__':
    args = parse_args()
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)

    # Init feature processor.
    root_dir = '/data/ainlp/classification/data/new_first_data'
    fp_obj = FeatureProcessor(root_dir,
                              ngram_range=(1, args.ngram_max),
                              feature_file=args.feature_dict)

    # Load training data and transform {{{.

    logging.info('Loading training data ...')

    train_titles, train_conts, train_labels = load_corpus(
        args.train_file, cols={'label':1, 'title':2, 'content':3})

    logging.info('Transforming training data ...')
    train_X = None

    if args.load_transform is not None:
        train_X = joblib.load(args.load_transform)
    else:
        train_X = fp_obj.transform(train_titles, train_conts)
        if args.save_transform is not None:
            joblib.dump(train_X,args.save_transform,compress=True)
    train_y = train_labels
    train_titles = None
    train_conts = None
    train_labels = None

    logging.info('Finished transforming, shape of train_X:' +
                 str(train_X.shape))
    # }}}.

    # Load test data and transform {{{.
    test_X = None
    test_y = None
    test_docs = None
    if args.test_file:
        logging.info('Loading test data ...')

        test_titles, test_conts, test_labels = load_corpus(
            args.test_file, cols={'label':1, 'title':2, 'content':3})
        test_X = fp_obj.transform(test_titles, test_conts)
        test_y = test_labels

        logging.info('Finished transforming, shape of test_X:' +
                     str(test_X.shape))

        test_docs = [' '.join([t, c]).strip() for t, c in
                     zip(test_titles, test_conts)]
        test_titles = None
        test_conts = None
        test_labels = None
    # }}}.

    # Train child models {{{.
    logging.info('Training models ...')

    # Create prefix of model name.
    model_prefix = args.feature_dict.split('/')[-1]
    if args.model_signature is not None:
        model_prefix = '%s.%s' % (model_prefix, args.model_signature)
    model_prefix = os.path.join(args.model_dir, model_prefix)

    # Train.
    train_models(train_X, train_y, test_X, test_y,
                 model_prefix=model_prefix)

    # Train fasttext.
    if args.ft_train_file:
        train_ft(args.ft_train_file, test_docs, test_y, model_prefix)

    ensemble = None
    #Train XGBOOST
    if args.train_xgboost == '1':
        if args.feature_mode == None:
            logging.error("XGBOOST NEED Feature Mode")
            sys.exit()
        logging.info('Loading training data ...')
        train_titles, train_conts, train_labels = load_corpus(
            args.train_file, cols={'label':1, 'title':2, 'content':3})

        train_docs = [' '.join([t, c]).strip() for t, c in
                     zip(train_titles, train_conts)]
        train_titles = None
        train_conts = None
        ensemble = Ensemble(args.feature_mode,args.feature_dict,model_prefix)
        ensemble.train_do(train_X,train_docs,train_labels)

    if args.test_file:
        if args.feature_mode == None:
            logging.error("XGBOOST NEED Feature Mode")
        if ensemble is None:
            ensemble = Ensemble(args.feature_mode,args.feature_dict,model_prefix)
        ensemble.test_do(test_X,test_docs,test_y)

    # }}}.

