#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors:  shuaijiang <shuaijiang@tencent.com>
# Create: 2018/08/05
import argparse
import codecs
from collections import defaultdict
import logging
import os
import sys

import fasttext
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from feature_processing import FeatureProcessor

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True,
                        help='segmented test data path')
    parser.add_argument('--feature_dict', type=str, required=True,
                        help='feature dict path')
    parser.add_argument('--test_ft', action='store_true', default=False,
                        help='feature dict path')
    parser.add_argument('--model_prefix', type=str, required=True,
                        help='prefix of model names')
    parser.add_argument('--ngram_max', type=int, default=2,
                        help='max value of ngram range')
    parser.add_argument('--label_out', type=str,
                        help='file to output predicted labels')
    parser.add_argument('--metrics_out', type=str,
                        help='csv files to output model metrics')
    return parser.parse_args()


def load_corpus(fname, cols={}):
    cmsid_col = cols.get('cmsid', -1)
    label_col = cols.get('label', -1)
    title_col = cols.get('title', -1)
    cont_col = cols.get('content', -1)
    max_col = max([label_col, title_col, cont_col])

    cmsids = []
    titles = []
    conts = []
    labels = []
    with open(fname) as fin:
        for i, line in enumerate(fin):
            if i % 10000 == 0:
                logging.info('Finished loading of %d lines' % i)

            fields = line.strip('\n').decode('utf8').split('\t')
            if max_col >= len(fields):
                continue

            if cmsid_col >= 0:
                cmsids.append(fields[cmsid_col])
            else:
                cmsids.append(None)

            if label_col >= 0:
                labels.append(fields[label_col])
            else:
                labels.append(None)

            if title_col >= 0:
                titles.append(fields[title_col])
            else:
                titles.append('')

            if cont_col >= 0:
                conts.append(fields[cont_col])
            else:
                conts.append('')
    return cmsids, labels, titles, conts


def save_metrics(metrics_file, y_gold, y_preds=[], descs=[]):
    '''Saving metrics result to a csv file.'''
    metrics = defaultdict(dict)
    columns = []
    labels = sorted(list(set(y_gold)))
    for y_pred, desc in zip(y_preds, descs):
        Ps, Rs, Fs, Supports = precision_recall_fscore_support(
            y_gold, y_pred, labels=labels)
        
        P_desc = '%s_P' % desc
        columns.append(P_desc)
        macro_avg = np.average(Ps)
        weight_avg = np.average(Ps, weights=Supports)
        metrics[P_desc] = Ps.tolist() + [macro_avg, weight_avg]

        R_desc = '%s_R' % desc
        columns.append(R_desc)
        macro_avg = np.average(Rs)
        weight_avg = np.average(Rs, weights=Supports)
        metrics[R_desc] = Rs.tolist() + [macro_avg, weight_avg]

        F_desc = '%s_F' % desc
        columns.append(F_desc)
        macro_avg = np.average(Fs)
        weight_avg = np.average(Fs, weights=Supports)
        metrics[F_desc] = Fs.tolist() + [macro_avg, weight_avg]

    rows = labels + ['Macro-avg', 'Weig-avg']
    metrics_df = pd.DataFrame(metrics, index=rows, columns=columns)
    metrics_df.to_csv(metrics_file)


def test_models(test_file, model_prefix, fp_obj, test_ft=False,
                label_out=None, metrics_out=None, report_out=sys.stdout):
    if fp_obj is None:
        raise ValueError('Feature processing object is None!')

    # Load text data.
    cmsids, labels, titles, contents = load_corpus(
        test_file, cols={'cmsid':0, 'label':1, 'title':2, 'content':3})
    labels = [l.replace('__label__', '') for l in labels]

    # Predict with ML models {{{.
    logging.info('Loading models ...')
    nb_model = joblib.load(model_prefix + '.nb')
    svm_model = joblib.load(model_prefix + '.svm')
    pa_model = joblib.load(model_prefix + '.pa')

    X = fp_obj.transform(titles, contents)
    nb_pred = [l.replace('__label__', '') for l in nb_model.predict(X)]
    pa_pred = [l.replace('__label__', '') for l in pa_model.predict(X)]
    svm_pred = [l.replace('__label__', '') for l in svm_model.predict(X)]
    # }}}.

    # Predict with fastText {{{.
    if test_ft:
        ft_model = fasttext.load_model(model_prefix + '.ft.bin')
        docs = []
        for title, cont in zip(titles, contents):
            doc = ' '.join([title, cont]).strip()
            docs.append(doc.encode('utf-8'))
        ft_pred = [x[0].replace('__label__', '')
                   for x in ft_model.predict(docs)]
    # }}}.

    # Output predicted labels {{{.
    if label_out is not None:
        if test_ft:
            label_out.write('cmsid\tgolden\tNB\tPA\tSVM\tFastText\n')
            for i, golden, nb_l, pa_l, svm_l, ft_l in zip(range(len(cmsids)),
														  labels,
														  nb_pred,
														  pa_pred,
														  svm_pred,
														  ft_pred):
                label_out.write('%s\n' % '\t'.join([cmsids[i],
                                                    golden,
                                                    nb_l,
                                                    pa_l,
                                                    svm_l,
                                                    ft_l]))
        else:
            label_out.write('cmsid\tgolden\tNB\tPA\tSVM\n')
            for i, golden, nb_l, pa_l, svm_l in zip(range(len(cmsids)),
													labels,
													nb_pred,
													pa_pred,
													svm_pred):
                label_out.write(
                    '%s\n' % '\t'.join([cmsids[i], golden, nb_l, pa_l, svm_l]))
    # }}}.

    # Output metrics {{{.
    if metrics_out is not None:
        if test_ft:
            save_metrics(metrics_out, labels,
                         [nb_pred, pa_pred, svm_pred, ft_pred],
                         ['NB', 'PA', 'SVM', 'FT'])
        else:
            save_metrics(metrics_out, labels,
                         [nb_pred, pa_pred, svm_pred],
                         ['NB', 'PA', 'SVM'])
    # }}}

    # Output evaluation reports {{{.
    if report_out is not None:
        print '==================NB_report:==================\n'
        print classification_report(labels, nb_pred, digits=4)

        print '=================PA_report:==================\n'
        print classification_report(labels, pa_pred, digits=4)

        print '=================SVM_report:==================\n'
        print classification_report(labels, svm_pred, digits=4)

        if test_ft and ft_pred is not None:
            print '=================FT_report:==================\n'
            print classification_report(labels, ft_pred, digits=4)
    # }}}.


if __name__ == '__main__':
    args = parse_args()

    # Init feature processor.
    root_dir = '/data/ainlp/classification/data/new_first_data'
    fp_obj = FeatureProcessor(root_dir,
                              ngram_range=(1, args.ngram_max),
                              feature_file=args.feature_dict)

    label_out = None
    if args.label_out is not None:
        label_out = codecs.open(args.label_out, 'w', encoding='utf8')
    metrics_out = None
    if args.metrics_out is not None:
        metrics_out = codecs.open(args.metrics_out, 'w', encoding='utf-8')

    # Test child models.
    test_models(args.test_file, args.model_prefix, fp_obj, args.test_ft,
				label_out=label_out, metrics_out=metrics_out)

    # close files
    if label_out is not None:
        label_out.close()
    if metrics_out is not None:
        metrics_out.close()

