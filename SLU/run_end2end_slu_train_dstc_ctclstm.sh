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
echo "Using fairseq-train: ${FAIRSEQ_PATH}/fairseq-train"
echo " ---"

# -----------------
XP_HEAD='DL-SLU'
FEATURES_HEAD='DSTC-TTS-All'
FEATURES_TYPE='FlowBERT'	# Use 'spectro', 'normspectro' or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' ... see below
FEATURES_SPEC='-xlsr53-56k' #'-16kHz.20.0ms-spg'
FEATURES_EXTN='.foo' #'.16kHz.20.0ms-spg'	# Feature file extension, e.g. '.20.0ms-spg', '.bert-3kb', '.bert-3kl', '.bert-3klt', ...
FEATURES_LANG='En'		# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
SUBTASK='concept'
CORPUS='fsc'
THEME='Default-tts_u-m_xlsr53_trs-ssl' # Default Context-Aware setting: RoBERTa Trs sum to speech input encoding (RoBERTaTrsSum; No decoder norm layers (NoDecNorm); Predicted decoder hidden state as context query (PredH); Use Context representation for initializing decoder hidden state (CtxHInit)
# -----------------

WORK_PATH=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/Dialog-Level-SLU/
DATA_PATH=${HOME}/work/data/DSTC/
SERIALIZED_CORPUS=${WORK_PATH}/system_features/${FEATURES_HEAD}.user+machine.${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}.data

# LSTMs in the encoder are bi-directional, thus the decoder size must be twice the size of the encoder
enc_type='ziggurat'
encoder_size=256    # 256+128 == 384
decoder_size=$((${encoder_size}))
ENC_FFN_DIM=$((${decoder_size}*4))
NCONV=1
conv_desc=""
if [[ ${enc_type} == 'deep-speech' ]]; then
	conv_desc="${NCONV}Convx"
fi
NLSTM=3
ATT_HEADS=8
DLAYERS=2
DROP_RATIO=0.35
TRANS_DROP_RATIO=0.25
MAX_EPOCHS=150
LR=0.0005
LR_SHRINK=0.98
WDECAY=0.0001
BATCH=16
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

ANNEAL=0
wu_epochs=6
curriculum=1
ctx_fusion='sum'
nbatches=7152	# WITH BATCH SIZE 16, User-only (no context): 3547; User+agent (no context): 6567; User+agent +context: 7152
wu_updates=$((${wu_epochs}*${nbatches}))	# Batch 5: MEDIA 5394, FSC 4627, PortMEDIA 2690, MEDIA+PortMEDIA 9275; Batch 10: MEDIA+PortMEDIA 5797
warmup_opt="--warmup-updates ${wu_updates}"
SAVE_PATH=${WORK_PATH}/XP/${XP_HEAD}_${FEATURES_HEAD}_${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}_${NLSTM}x${conv_desc}Enc-${enc_type}-H${encoder_size}-${DLAYERS}xDec-${DECODER}-${SUBTASK}-H${decoder_size}-Drop-Enc${DROP_RATIO}-Dec${TRANS_DROP_RATIO}_BATCH${BATCH}_LR${LR}_WU${wu_updates}_${CRITERION}${LABEL_SMOOTH}_${THEME}_Curric${curriculum}_NEWTEST/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

reg_options="--clip-norm ${CLIP_NORM} --weight-decay ${WDECAY}"

ROBERTA_MODEL='/home/getalp/dinarelm/work/data/models/Roberta/roberta.large/model.pt'
D_AND_R_MODEL='/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/Dialog-Level-SLU/XP/DL-SLU_DSTC_D-and-R_FlowBERT-xlsr53-56k-En_3xEnc-ziggurat-H256-2xDec-ctclstm-concept-H256-Drop-Enc0.35-Dec0.25_BATCH16_LR0.0005_WU42912_slu_ctc_loss_1Step_RoBERTaTrsSum_SumCtx4-Disc0.5_NoDecNorm_PredH_CtxHInit_Curric1_TEST02/checkapoint.ensemble.pt'
DSTC_EXTRA_DATA='/home/getalp/dinarelm/work/data/DSTC/train.100x.tts-verbatim.transcript.2022-10-17.txt'
AUX_OPTIONS="${AUX_OPTIONS} --use-transcription-as-input --dialog-level-slu --normalize-dialog-batches --use-dialog-history --context-fusion ${ctx_fusion} --context-size 4 --context-first-turns 0 --context-discount 0.5 --roberta-model ${ROBERTA_MODEL}" #--model-start-point ${D_AND_R_MODEL}"

GPU_ID=1
#CUDA_VISIBLE_DEVICES=${GPU_ID}
PYTHONPATH=${HOME}/work/tools/fairseq/ ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} \
	--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 --feature-extension ${FEATURES_EXTN} \
	--decoder ${DECODER} --padded-reference --corpus-name ${CORPUS} ${AUX_OPTIONS} \
	--save-dir ${SAVE_PATH} --patience 20 --no-epoch-checkpoints \
	--speech-encoder ${enc_type} --speech-conv ${NCONV} --num-features ${NUM_FEATURES} --speech-conv-size ${encoder_size} --drop-ratio ${DROP_RATIO} \
	--num-lstm-layers ${NLSTM} --speech-lstm-size ${encoder_size} --window-time ${WINTIME} --w2v-language ${FEATURES_LANG} \
	--encoder-normalize-before --encoder-layers 1 --encoder-attention-heads ${ATT_HEADS} --encoder-ffn-embed-dim ${ENC_FFN_DIM} \
	--decoder-normalize-before --attention-dropout ${TRANS_DROP_RATIO} --activation-dropout ${TRANS_DROP_RATIO} \
	--decoder-layers ${DLAYERS} --share-decoder-input-output-embed \
	--encoder-embed-dim ${encoder_size} --encoder-hidden-dim ${encoder_size} --decoder-embed-dim ${decoder_size} --decoder-hidden-dim ${decoder_size} \
	--dropout ${TRANS_DROP_RATIO} --decoder-dropout ${TRANS_DROP_RATIO} --decoder-attention-heads ${ATT_HEADS} \
	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} \
	--lr-scheduler fixed --lr ${LR} --force-anneal ${ANNEAL} --lr-shrink ${LR_SHRINK} ${warmup_opt} \
	--optimizer adam ${reg_options} \
	--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum ${curriculum} --keep-best-checkpoints 6

