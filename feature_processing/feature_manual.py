# coding:utf-8
import numpy as np
import scipy.sparse as sp

from feature_base import FeatureBase


class ManualFeature(FeatureBase):
    def __init__(self, feature_prefix=''):
        super(ManualFeature, self).__init__(feature_prefix)
        self.feature_prefix = feature_prefix
        self.manual_feature_config = [
            ('manual_feature_1', self._manual_feature_1),
        ]
        # Create feature vocabulary.
        for fea_name, fea_proc in self.manual_feature_config:
            if feature_prefix:
                fea_name = '%s_%s' % (feature_prefix, fea_name)
            self.vocabulary_[fea_name]

    def _manual_feature_1(self, title, cont):
        ''' Manual feature indicates if specific keywords exist in
        title or not, keywords include "神剧", "电视剧".
        '''
        keywords = set([u'神剧', u'电视剧'])
        title_words = set(title.split(' '))
        if keywords & title_words:
            return 1
        else:
            return 0

    def transform_one(self, title, cont):
        feature_counter = {}
        for fea_name, fea_proc in self.manual_feature_config:
            if self.feature_prefix:
                fea_name = '%s_%s' % (self.feature_prefix, fea_name)

            if fea_name in self.vocabulary_:
                fea_value = fea_proc(title, cont)
                feature_counter[fea_name] = fea_value
        return feature_counter

    def transform(self, titles, conts):
        if not isinstance(titles, list) or not isinstance(conts, list):
            raise ValueError('List of doc string expected.')

        if len(titles) != len(conts):
            raise ValueError('Docs and titles must have the same length.')

        values = []
        j_indices = []
        indptr = [0]
        for title, cont in zip(titles, conts):
            feature_counter = {}
            for fea_name, fea_proc in self.manual_feature_config:
                full_fea_name = fea_name
                if self.feature_prefix:
                    full_fea_name = '%s_%s' % (self.feature_prefix, fea_name)

                if full_fea_name not in self.vocabulary_:
                    continue

                fea_idx = self.vocabulary_[full_fea_name]
                fea_value = fea_proc(title, cont)
                feature_counter[fea_idx] = fea_value

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        j_indices = np.asarray(j_indices, dtype=np.int32)
        indptr = np.asarray(indptr, dtype=np.int32)
        values = np.asarray(values, dtype=np.int32)
        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr)-1, len(self.vocabulary_)),
                          dtype=np.int32)
        X.sort_indices()
        return X


if __name__ == '__main__':
    mf_obj = ManualFeature()
    titles = ['抗日 神剧 相信']
    conts = ['正文 内容 大结局 雷人']
    x = mf_obj.transform(titles, conts)
    print x.toarray()
