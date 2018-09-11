# coding:utf-8
from collections import defaultdict
import logging
import numbers
import sys

import numpy as np
import scipy.sparse as sp

from feature_base import FeatureBase

reload(sys)
sys.setdefaultencoding('utf-8')

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format=('[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d :'
            '%(message)s'), datefmt='%a, %d %b %Y %H:%M:%S')

chinese_tokeizer = lambda s:s.split()

class NgramFeature(FeatureBase):
    def __init__(self, ngram_range=(1,1), feature_prefix='',
                 tokenizer=chinese_tokeizer, stop_words=None):
        super(NgramFeature, self).__init__(feature_prefix)
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def fit(self, docs, encoding='utf-8', max_df=1.0, min_df=1,
            max_features = None):
        # Count raw ngrams.
        n_doc, ngram_df_dict = self._count_ngrams(docs, encoding)
        # Compute limit numbers
        max_doc_count = (max_df if isinstance(max_df, numbers.Integral)
                         else max_df*n_doc)
        min_doc_count = (min_df if isinstance(min_df, numbers.Integral)
                         else min_df*n_doc)
        if max_doc_count < min_doc_count:
            raise ValueError('max_df corresponds to < documents than min_df')
        # compute limit feature mask
        limit_mask = self._limit_feature_mask(ngram_df_dict, max_doc_count,
                                              min_doc_count, max_features)
        if len(ngram_df_dict) != len(limit_mask):
            raise ValueError('Mismatch length of features and mask')
        # update feature vocabulary
        self.vocabulary_ = defaultdict()
        self.vocabulary_.default_factory = self.vocabulary_.__len__
        for i, (feature, df) in enumerate(ngram_df_dict.items):
            if not limit_mask[i]:
                continue
            if self.feature_prefix:
                feature = '%s_%s' % (self.feature_prefix, feature)
            self.vocabulary_[feature]
        logging.info('Get %d ngram features after limit filtering.'
                     % len(self.vocabulary_))

    def _count_ngrams(self, raw_docs, encoding=None):
        ngram_df_dict = {}
        for i,doc in enumerate(raw_docs):
            if i % 10000 == 0:
                logging.info('Finished loading of %d lines' % i)
            ngrams_set = set(self.get_ngrams(doc))
            for feature in ngrams_set:
                ngram_df_dict[feature] = ngram_df_dict.get(feature,0) + 1
        logging.info('Finished count ngrams in corpus, totally %d docs,'
                     '%d uniq ngrams.' % (i, len(ngram_df_dict)))
        return i, ngram_df_dict

    def get_ngrams(self, doc, encoding=None, prefix=False):
        doc = doc.strip()
        if not doc:
            return []
        if not isinstance(doc, unicode):
            if encoding is None:
                raise ValueError('Docs should be Unicode or encoding is provided')
            else:
                doc = doc.decode(encoding)
        if self.tokenizer:
            tokens = self.tokenizer(doc)
        else:
            tokens = doc.split()
        if prefix:
            new_tokens = []
            for ngram in self._word_ngrams(tokens):
                feature = '%s_%s' % (self.feature_prefix, ngram)
                new_tokens.append(feature)
            return new_tokens
        else:
            return self._word_ngrams(tokens)

    def _word_ngrams(self, tokens):
        '''Turn tokens into a sequences of n-gram after stop words filtering'''
        # Handle stop words
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]
        # Handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []
            n_original_tokens = len(original_tokens)
            # Bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = ' '.join
            for n in xrange(min_n, min(max_n+1, n_original_tokens+1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i:i+n]))
        return tokens

    def _limit_feature_mask(self, df_dict, high=None, low=None, limit=None):
        """Remove too rare or too common features"""
        dfs = np.array(df_dict.values())
        mask = np.ones(len(dfs), dtype=bool)
        if high is None and low is None and limit is None:
            return mask
        if high is not None:
            mask &= (dfs <= high)
        if low is not None:
            mask &= (dfs >= low)
        if limit is not None and mask.sum() > limit:
            mask_inds = (-dfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask
        return mask
