# coding:utf-8

from collections import defaultdict

class FeatureBase(object):
    """Base class for feature extraction"""
    def __init__(self, feature_prefix=''):
        self.feature_prefix = feature_prefix
        self.vocabulary_ = defaultdict()
        self.vocabulary_.default_factory = self.vocabulary_.__len__

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name."""
        return [t for t,i in sorted(self.vocabulary_.items(), key=lambda x:x[1])]
