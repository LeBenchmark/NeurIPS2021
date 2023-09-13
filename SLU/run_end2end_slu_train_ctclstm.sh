#!/bin/bash

# TO RUN ON DECORE AND OTHER SERVERS UNCOMMENT THE FOLLOWING LINES
#export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${HOME}/anaconda3/

# TO RUN ON THE LIG GRID WITH OAR UNCOMMENT THE FOLLOWING LINES
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#PYTHONPATH=${HOME}/work/tools/fairseq/
#export RNNTAGGERPATH=${HOME}/work/tools/Seq2Biseq_End2EndSLU/

echo " ---"
echo "Using python: `which python`"
echo " ---"

RUN_TRAIN=1
# -----------------
XP_HEAD='FlowBERT-Journal'
FEATURES_HEAD='MEDIA'
FEATURES_TYPE='FlowBERT'	# Use 'spectro', 'normspectro' or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' ... see below
FEATURES_SPEC='-w2v2-7kl-fs-char-tuned-old-fix' #'-w2v2-7kl-fs-char-tuned-old-fix' #'-w2v2-7kl-fs-char-tuned-old-fix' # Nest xp: -w2v2-14kl-sb-2-char-tuned-old-fix, -w2v2-14kxl-sb-char-tuned-old-fix
FEATURES_EXTN='.foo'	# Feature file extension, e.g. '.20.0ms-spg', '.bert-3kb', '.bert-3kl', '.bert-3klt', ...
FEATURES_LANG='Fr'		# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
SUBTASK='concept'
CORPUS='media'
THEME='Default_mt-loss.75_mha2'
# -----------------

WORK_PATH=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/
PREFIX_PATH=${WORK_PATH}/works/2024_LREC-Coling_DL-SLU/
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogSemTextAndWav_FixedChannel_EnContexte/ #DialogMachineAndUser_SemTextAndWav_FixedChannel/
#DATA_PATH=${HOME}/work/data/FluentSpeechCommands/fluent_speech_commands_dataset/wavs/
#DATA_PATH=${HOME}/work/data/PortMEDIA/DialogTextAndWav/
#DATA_PATH=${HOME}/work/data/ETAPE/
SERIALIZED_CORPUS=${WORK_PATH}/system_features/${FEATURES_HEAD}.user+machine.${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}.data

# LSTMs in the encoder are bi-directional, thus the decoder size must be twice the size of the encoder
enc_type='ziggurat'
encoder_size=256
decoder_size=$((${encoder_size}))
ENC_FFN_DIM=$((${decoder_size}*4))
NCONV=1
conv_desc=""
if [[ ${enc_type} == 'deep-speech' ]]; then
	conv_desc="${NCONV}Convx"
fi
NLSTM=3
ATT_HEADS=4
DLAYERS=2
DROP_RATIO=0.35
TRANS_DROP_RATIO=0.25
MAX_EPOCHS=150
LR=0.0005
LR_SHRINK=0.98
WDECAY=0.0001
UPDATE_FREQ=1
BATCH=5
MAX_TOKENS=4750
# ----------------
NUM_FEATURES=81
WINTIME=20
if [[ ${FEATURES_TYPE} == 'normspectro' ]]; then
        NUM_FEATURES=1025
fi
if [[ ${FEATURES_TYPE} == 'W2V' ]]; then
        NUM_FEATURES=512
        WINTIME=-1
fi
if [[ ${FEATURES_TYPE} == 'W2V2-Large' ]]; then
        NUM_FEATURES=1024
        WINTIME=-6
fi
if [[ ${FEATURES_TYPE} == 'W2V2' ]]; then
        NUM_FEATURES=768
        WINTIME=-4
fi
if [[ ${FEATURES_TYPE} == 'FlowBERT' ]]; then
        NUM_FEATURES=1024
        WINTIME=-3
fi
if [[ ${FEATURES_TYPE} == 'FlowBBERT' ]]; then
        NUM_FEATURES=768
        WINTIME=-5
fi
if [[ ${FEATURES_TYPE} == 'XLSR53' ]]; then
        NUM_FEATURES=1024
        WINTIME=-7
fi
# ----------------
CLIP_NORM=5.0
DECODER='ctclstm'
CRITERION='slu_ctc_loss' # Use 'cross_entropy'  or 'label_smoothed_cross_entropy' for Cross-Entropy loss, 'slu_ctc_loss' for CTC loss
LABEL_SMOOTH=""
AUX_OPTIONS=""
if [[ ${CRITERION} == "label_smoothed_cross_entropy" ]]; then
	LABEL_SMOOTH=0.001
	AUX_OPTIONS="--label-smoothing ${LABEL_SMOOTH}"
fi

if [[ $# -ge 1 ]]; then
	LR=$1
fi

ANNEAL=1
wu_epochs=6	# NOTE: set this to 6 for fine-tuned SSL models and Distilled models, 8 for fine-tuned models when using also transcription input encoded with ssl (i.e. w2v2 + CamemBERT)
curriculum=1
ctx_fusion='sum'
nbatches=5394	# 6741 # Batch 5: MEDIA user turns only: 2758; MEDIA normalized for dialog-level SLU: 5614
wu_updates=$((${wu_epochs}*${nbatches}))	# Batch 5: MEDIA 5394, FSC 4627, PortMEDIA 2690, MEDIA+PortMEDIA 9275; Batch 10: MEDIA+PortMEDIA 5797
warmup_opt="--warmup-updates ${wu_updates}"
reg_options="--clip-norm ${CLIP_NORM} --weight-decay ${WDECAY}"

BASE_OPTIONS="${AUX_OPTIONS}"
IN_TRS_OPTIONS="--use-transcription-as-input --roberta-model ${HOME}/work/data/models/Roberta/camembert-base/"
DIALOG_LEVEL_OPTIONS="${AUX_OPTIONS} --dialog-level-slu --normalize-dialog-batches --use-dialog-history --context-fusion ${ctx_fusion} --context-size 6 --context-first-turns 0"
ONLINE_FEATURES_OPTIONS="${AUX_OPTIONS} --online-feature-extraction --upsample --feature-extractor whisper"
#NUM_FEATURES=768	# when extracting features online, this must correspond to the actual size of features extracted with the feature extractor !
ASR_SLU_LOSS_OPTIONS="${AUX_OPTIONS} --asr-slu-loss"
#enc_type="sb-wav2vec2"
SBWAV2VEC2_OPTIM_OPTIONS="${AUX_OPTIONS} --speech-encoder ${enc_type} --upsample --se-size 1024 --sb-model-path ${HOME}/work/data/models/wav2vec2.0/wav2vec2-large-14k-sb-2/ --max-source-positions 400000"

RISE_TEMP_EPOCH=60
RISE_TEMP_STRATEGY='linear'
START_TEMP=2.0
END_TEMP=4.0
RISE_TEMP_OPTIONS="--rise-temperature-at-epoch ${RISE_TEMP_EPOCH} --rise-temperature-strategy ${RISE_TEMP_STRATEGY} --softmax-start-temperature ${START_TEMP} --softmax-temperature=${END_TEMP}"
ADDITIONAL_OPTIONS="--attention-type mha2"

SAVE_PATH=${PREFIX_PATH}/${XP_HEAD}_${FEATURES_HEAD}_${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}_${NLSTM}x${conv_desc}Enc-${enc_type}-H${encoder_size}-${DLAYERS}xDec-${DECODER}-${SUBTASK}-H${decoder_size}-Drop-Enc${DROP_RATIO}-Dec${TRANS_DROP_RATIO}_BATCH${BATCH}_LR${LR}_WU${wu_updates}_${CRITERION}${LABEL_SMOOTH}_${THEME}_Curriculum${curriculum}_Clip${CLIP_NORM}_TEST/ #SoftT@${RISE_TEMP_STRATEGY}${RISE_TEMP_EPOCH}-${START_TEMP}-to-${END_TEMP}_TEST/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

GPU_ID=1
if [[ ${RUN_TRAIN} -eq 1 ]]; then
	#CUDA_VISIBLE_DEVICES=${GPU_ID}
	PYTHONPATH=${HOME}/work/tools/fairseq/ ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} ${ADDITIONAL_OPTIONS} --update-freq ${UPDATE_FREQ} \
		--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 --feature-extension ${FEATURES_EXTN} \
		--decoder ${DECODER} --padded-reference --corpus-name ${CORPUS} \
		--save-dir ${SAVE_PATH} --patience 20 --no-epoch-checkpoints \
		--speech-encoder ${enc_type} --speech-conv ${NCONV} --num-features ${NUM_FEATURES} --speech-conv-size ${encoder_size} --drop-ratio ${DROP_RATIO} \
		--num-lstm-layers ${NLSTM} --speech-lstm-size ${encoder_size} --window-time ${WINTIME} --w2v-language ${FEATURES_LANG} \
		--encoder-normalize-before --encoder-layers 1 --encoder-attention-heads ${ATT_HEADS} --encoder-ffn-embed-dim ${ENC_FFN_DIM} \
		--decoder-normalize-before --attention-dropout ${TRANS_DROP_RATIO} --activation-dropout ${TRANS_DROP_RATIO} \
		--decoder-layers ${DLAYERS} --share-decoder-input-output-embed \
		--encoder-embed-dim ${encoder_size} --encoder-hidden-dim ${encoder_size} --decoder-embed-dim ${decoder_size} --decoder-hidden-dim ${decoder_size} \
		--output-size=$((2*${decoder_size})) --dropout ${TRANS_DROP_RATIO} --decoder-dropout ${TRANS_DROP_RATIO} --decoder-attention-heads ${ATT_HEADS} \
		--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} \
		--lr-scheduler fixed --lr ${LR} --force-anneal ${ANNEAL} --lr-shrink ${LR_SHRINK} ${warmup_opt} \
		--optimizer adam ${reg_options} \
		--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum ${curriculum} --keep-best-checkpoints 5
fi

# Evaluation
echo " ***"
echo " Running evaluation..."
echo " ***"
EVAL_SCRIPT=${HOME}/work/tools/fairseq_tools/end2end_slu/scripts/run_end2end_slu_test.sh
${EVAL_SCRIPT} ${SAVE_PATH}/checkpoint_best.pt ${SERIALIZED_CORPUS} concept "${ADDITIONAL_OPTIONS}"

