#!/usr/bin/sh
# offline training scripts


RAW_TRAIN_FILE=""
SEG_TRAIN_FILE="./refine/local_data/cat_train_v2.2.seg20180713"
FT_SEG_TRAIN_FILE="./refine/local_data/cat_train_v2.2.seg20180713.ft"
SEG_TEST_FILE="./refine/local_data/cat_test_v1.seg20180712"

MAX_FEATURES=500000
NGRAM_MAX=1
FEATURE_DICT="feature/cat_train_v2.2.seg20180713.fea_ug_50w"
FEATURE_LOG="logs/feature.train_v2.2.fea_ug_50w.log"
TRAIN_LOG="logs/train.v2.2.fea_ug_50w.log"

PrintLog() {
    time_str=`date +%Y-%m-%d,%T`
    echo "[${time_str}] $1"
}

# TODO: segment the raw training corpus if need {{{.
#python segment_training_data.py --train_file TRAIN_FILE --seg_train_file SEG_TRAIN_FILE
# }}}.

# Extract and update feature dict {{{.
PrintLog "Creating feature dict (${FEATURE_DICT}) ..."
python ./create_feature_dict.py --train_file ${SEG_TRAIN_FILE} \
    --feature_dict ${FEATURE_DICT} --max_features ${MAX_FEATURES} \
    --ngram_max ${NGRAM_MAX} > ${FEATURE_LOG} 2>&1
if [ $? -ne 0 ];then
    echo "Failed creating feature dict!"
    exit 1
fi
# }}}.

# Train models {{{.
PrintLog "Training models ..."
PrintLog "[corpus] ${SEG_TRAIN_FILE}"
PrintLog "[feature] ${FEATURE_DICT}"
python ./train.py --train_file ${SEG_TRAIN_FILE} --test_file ${SEG_TEST_FILE} \
    --feature_dict ${FEATURE_DICT} --ft_train_file ${FT_SEG_TRAIN_FILE} \
    --ngram_max ${NGRAM_MAX} > ${TRAIN_LOG} 2>&1
if [ $? -ne 0 ];then
    echo "Failed training!"
    exit 1
fi
# }}}.

PrintLog "finished!"
