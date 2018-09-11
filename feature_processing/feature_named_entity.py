# coding:utf-8
import numpy as np
import scipy.sparse as sp

from feature_base import FeatureBase

class NameEntityFeature(FeatureBase):
    def __init__(self, entity_info_file, channel_def_file,
                 entity_type_file, feature_prefix=''):
        super(NameEntityFeature, self).__init__(feature_prefix)
        self._load_entity_data(entity_info_file, channel_def_file, entity_type_file)
        # Create feature vocabulary
        for channel_etype in self.channel_etype_dict:
            feature_name = channel_etype
            if feature_prefix:
                feature_name = '%s_%s' % (feature_prefix, channel_etype)
            self.vocabulary_[feature_name]

    def _load_entity_data(self, entity_info_file, channel_def_file,
                          entity_type_file):
        # Mapping from channel id to channel name
        self.channel_map = self._load_channels(channel_def_file)
        # Mapping from entity type id to entity type name
        self.etype_map = self._load_entity_types(entity_type_file)
        # Mapping from entity to channel-to-entity_type(e.g:
        # "花千骨" -> "娱乐_电视剧").
        entity_ch_etype_map = {}
        with open(entity_info_file) as fin:
            for i,line in enumerate(fin):
                fields = line.decode('utf-8').split('\t')
                if len(fields) < 3:
                    continue
                channel_id = fields[0].strip()
                channel = self.channel_map.get(channel_id, None)
                if not channel:
                    continue
                entity = fields[1].strip().lower()
                if not entity or len(entity) < 2:
                    continue
                etype_ids = fields[2].split(',')
                if not etype_ids or etype_ids[0] not in self.etype_map:
                    continue
                etype = self.etype_map[etype_ids[0]]
                channel_etype = '%s_%s'%(channel, etype)
                v_set = entity_ch_etype_map.setdefault(entity, set())
                v_set.add(channel_etype)
        # Mapping from entity to channel-to-entity_type for unambiguous
        # entities which means has uniq channel-entity_type.
        # Count occurrence of each channel_etype and remove entities
        # with multiple channel_etype which means this entity is
        # ambiguous.
        self.channel_etype_dict = {}
        self.entity_channel_etype_map = {}
        for entity, v_set in entity_ch_etype_map.items():
            if len(v_set) == 1:
                ch_etype = list(v_set)[0]
                self.entity_channel_etype_map[entity] = ch_etype
                self.channel_etype_dict[ch_etype] = self.channel_etype_dict.get(
                    ch_etype, 0) + 1

    def _load_channels(self, fname):
        id_channel_dict = {}
        with open(fname) as fin:
            for line in fin:
                fields = line.strip('\n').decode('utf-8').split('\t')
                if len(fields) < 3 or not fields[0]:
                    continue
                _id = fields[0].strip()
                channel = fields[2].strip()
                id_channel_dict[_id] = channel
        return id_channel_dict

    def _load_entity_types(self, fname):
        id_type_dict = {}
        with open(fname) as fin:
            for line in fin:
                fields = line.strip().decode('gbk').split('\t')
                if len(fields) < 2 or not fields[0]:
                    continue
                _id = fields[0].strip()
                etype = fields[1].strip()
                id_type_dict[_id] = etype
        return id_type_dict

    def transform_one(self, doc):
        feature_counter = {}
        for term in doc.split():
            if term not in self.entity_channel_etype_map:
                continue
            ch_etype = self.entity_channel_etype_map[term]
            if self.feature_prefix:
                fea_name = '%s_%s' % (self.feature_prefix, ch_etype)
            else:
                fea_name = ch_etype
            if fea_name in self.vocabulary_:
                feature_counter[fea_name] = feature_counter.get(
                    fea_name, 0) + 1
        return feature_counter


    def transform(self, docs):
        if not isinstance(docs, list):
            raise ValueError('List of list of terms over docs expected.')
        values = []
        j_indices = []
        indptr = [0]
        for doc in docs:
            feature_counter = {}
            for term in doc.split():
                if term not in self.entity_channel_etype_map:
                    continue
                ch_etype = self.entity_channel_etype_map[term]
                ch_etype = '%s_%s' % (self.feature_prefix, ch_etype)
                if ch_etype not in self.vocabulary_:
                    continue
                fea_idx = self.vocabulary_[ch_etype]
                feature_counter[fea_idx] = feature_counter.get(fea_idx, 0) + 1
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
