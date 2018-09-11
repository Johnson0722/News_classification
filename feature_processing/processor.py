# coding:utf-8
import codes
from collections import defaultdict
import logging
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2

from feature_manual import ManualFeature
from feature_named_entity import NamedEntityFeature
from feature_ngram import NgramFeature

reload(sys)
sys.setdefaultencodeing('utf-8')

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

chinese_tokeizer = lambda s:s.split()

class FeatureProcessor(object):
    def __init__(self, feature_data_root, feature_file=None, rebuild_feature=None,
                 tfidf_model=None, use_ne_norm=True, use_idf=True,
                 ngram_range=(1,2), word_fea_prefix='WRD',
                 entity_norm_fea_prefix='NEN', manual_fea_prefix='MAN',
                 idf_norm='l2'):
        self.ngram_range = ngram_range
        self.word_fea_prefix = word_fea_prefix
        # load stop words
        self.stop_words = set()
        stop_words_file = os.path.join(feature_data_root, 'stop_words.dat')
        self._load_stop_words(stop_words_file)
        # Init ngram feature object
        self.ngram_feature_obj = NgramFeature(ngram_range=ngram_range,
                                              feature_prefix=word_fea_prefix,
                                              stop_words=self.stop_words)
        # Init named entity normalization feature object
        self.use_ne_norm = use_ne_norm
        self.ne_feature_obj = None
        if use_ne_norm:
            self.ne_feature_obj = NamedEntityFeature(
                os.path.join(feature_data_root, 'tag_entity.dat'),
                os.path.join(feature_data_root, 'channel_map.dat'),
                os.path.join(feature_data_root, 'schema.dat'),
                feature_prefix=entity_norm_fea_prefix)
        # Init manual feature object
        self.mn_feature_obj = ManualFeature(feature_prefix=manual_fea_prefix)
        # Init empty feature vocabulary
        self.vocabulary_ready_ = False
        self.vocabulary_ = defaultdict()
        self.vocabulary_.default_factory = self.vocabulary_.__len__
        # Init feature vocabulary from feature file
        if feature_file:
            vocab_dict = self._load_feature_dict(feature_file)
            if rebuild_feature:
                self.rebuild_feature_dict(vocab_dict)
            else:
                self.vocabulary_ = vocab_dict
            self.vocabulary_ready_ = True

    def _load_feature_dict(self, feature_file):
        if not os.path.isfile(feature_file):
            raise ValueError('File %s not exist' % feature_file)
        vocabulary = dict()
        with open(feature_file) as fin:
            for line in fin:
                parts = line.strip().decode('utf8').split('\t')
                if len(parts) != 2:
                    continue
                fea_name = parts[0].strip()
                fea_idx = int(parts[1].strip())
                vocabulary[fea_name] = fea_idx
        logging.info('Finished loading feature dict from file, '
                     'totally %d features.' % len(vocabulary))
        return vocabulary

    def _load_stop_words(self, stop_words_file):
        with open(stop_words_file) as fin:
            for line in fin:
                word = line.strip().decode('utf-8')
                self.stop_words.add(word)
        logging.info('Finished loading stop words, totally %d ones'
                     % len(self.stop_words))

    def fit(self, docs, encoding='utf-8', max_df=1.0, min_df=1,
            max_features=None):
        # Fit ngram vectorizer.
        self.ngram_feature_obj.fit(
            docs, encoding='utf-8', max_df=max_df, min_df=min_df,
            max_features=max_features)
        # Feature selection to generate feature dict.
        self.feature_selection()

    def rebuild_feature_dict(self, old_vocab):
        # Fill extra feature names
        self._select_extra_feature()
        # Fill ngram feature names
        sorted_vocab = sorted(old_vocab.items(), key=lambda x:x[1])
        for feature_name, _ in sorted_vocab:
            if feature_name.startswith(self.word_fea_prefix):
                self.vocabulary_[feature_name]

    def feature_selection(self):
        self._select_extra_feature()
        # TODO: implement different feature selection methods on ngram features.
        self._default_ngram_feature_selection()
        # Set vocabulary done flag.
        self.vocabulary_ready_ = True

    def _select_extra_feature(self):
        # Get manual feature.
        if self.mn_feature_obj:
            for feature_name in self.mn_feature_obj.get_feature_names():
                self.vocabulary_[feature_name]
        # Get named entity normalization feature.
        if self.use_ne_norm and self.ne_feature_obj:
            for feature_name in self.ne_feature_obj.get_feature_names():
                self.vocabulary_[feature_name]

    def _default_ngram_feature_selection(self):
        for feature_name in self.ngram_feature_obj.get_feature_names():
            fea_name = feature_name
            self.vocabulary_[fea_name]

    def get_feature_names(self):
        '''Array mapping from feature integer indices to feature name.'''
        return [t for t, i in sorted(self.vocabulary_.items(),
                                     key=lambda x:x[1])]

    def dump_features(self, fname):
        if not self.vocabulary_ready_:
            return
        with codecs.open(fname, 'w', 'utf8') as fout:
            for fea, idx in sorted(self.vocabulary_.items(),
                                   key=lambda x:x[1]):
                fout.write('%s\t%s\n' % (fea, idx))

    def transform(self, titles, conts):
        if not isinstance(titles, list) or not isinstance(conts, list):
            raise ValueError('List of doc string expected.')
        if len(titles) != len(conts):
            raise ValueError('Docs and titles must have the same length.')
        if not self.vocabulary_ready_:
            raise ValueError('Feature vocabulary not initiaized yet!'
                             'Can not do transforming.')
        # Re-construct feature vector.
        values = []
        j_indices = []
        indptr = [0]
        for i in range(len(titles)):
            if i % 10000 == 0:
                logging.info('Finished transforming %d lines' % i)
            title = titles[i]
            cont = conts[i]
            feature_counter = {}
            # Fill n-gram features.
            ngrams = (self.ngram_feature_obj.get_ngrams(title, prefix=True) +
                      self.ngram_feature_obj.get_ngrams(cont, prefix=True))
            for fea_name in ngrams:
                if fea_name in self.vocabulary_:
                    fea_idx = self.vocabulary_[fea_name]
                    feature_counter[fea_idx] = feature_counter.get(
                        fea_idx, 0) + 1
            # Fill named entity normalization features.
            doc = ' '.join([title, cont]).strip()
            ne_features = self.ne_feature_obj.transform_one(doc)
            for fea_name, fea_freq in ne_features.items():
                if fea_name in self.vocabulary_:
                    fea_idx = self.vocabulary_[fea_name]
                    feature_counter[fea_idx] = feature_counter.get(
                        fea_idx, 0) + fea_freq
            # Fill manual features.
            mn_features = self.mn_feature_obj.transform_one(titles[i], conts[i])
            for fea_name, fea_freq in mn_features.items():
                if fea_name in self.vocabulary_:
                    fea_idx = self.vocabulary_[fea_name]
                    feature_counter[fea_idx] = feature_counter.get(
                        fea_idx, 0) + fea_freq
            # Update csr_matrix data.
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))
        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.asarray(indptr, dtype=np.intc)
        values = np.asarray(values, dtype=np.float64)
        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr)-1, len(self.vocabulary_)))
        X.sort_indices()
        inplace_csr_row_normalize_l2(X)
        return X


if __name__ == '__main__':
    root_dir = '/data/ainlp/classification/data/new_first_data'
    titles = [u'抗日 神剧 历史 琅琊榜 灵魂摆渡 泰坦尼克号 还珠格格',
              u'电影 aaa',
              u'社会 tianqi']
    conts = [u'我 是 中国 人 ， 我 爱 中国 。',
             u'我 喜欢 中国 ， 欢迎',
             u'中国 中国 中国 很 美丽 。']
    docs = [' '.join([t, c]).strip() for t, c in zip(titles, conts)]

    # Test fit feature_dict {{{.
    fp_obj = FeatureProcessor(root_dir)
    fp_obj.fit(docs)
    fp_obj.dump_features('feature_map.txt')
    # }}}.

    # Test load feature_dict and transform {{{.
    fp_obj = FeatureProcessor(root_dir, 'feature_map.txt')
    X = fp_obj.transform(titles, conts)
    for i in range(X.shape[0]):
        print '==> doc %d' % i
        row = X.getrow(i)
        print row.indptr
        print row.indices
        print row.data
    # }}}.

