#!/bin/bash

# TO RUN ON DECORE AND OTHER SERVERS UNCOMMENT THE FOLLOWING LINES
#export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${HOME}/anaconda3/

# TO RUN ON THE LIG GRID WITH OAR UNCOMMENT THE FOLLOWING LINES
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/

echo " ---"
echo "Using python: `which python`"
echo "Using fairseq-train: `which fairseq-train`"
echo " ---"

# -----------------
FEATURES_HEAD='MEDIA'
FEATURES_TYPE='FlowBERT'	# Use 'spectro', 'normspectro' or 'W2V' or 'W2V2' or 'FlowBERT' or 'FlowBBERT' ... see below
FEATURES_SPEC='-7kl-split'		# Choose a meaningful infix to add into file names, you can leave it empty for spectrograms (*-spg)
FEATURES_EXTN='.w2v2-fr-7kl'	# Feature file extension, e.g. '.20.0ms-spg', '.bert-3kb', '.bert-3kl', '.bert-3klt', ...
FEATURES_LANG='Fr'		# Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
SUBTASK='concept'
CORPUS='media'
# -----------------
WORK_PATH=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
#DATA_PATH=${HOME}/work/data/FluentSpeechCommands/fluent_speech_commands_dataset/wavs/
#DATA_PATH=${HOME}/work/data/PortMEDIA/DialogTextAndWav/
#DATA_PATH=${HOME}/work/data/ETAPE/
SERIALIZED_CORPUS=${WORK_PATH}/system_features/${FEATURES_HEAD}.user+machine.${FEATURES_TYPE}${FEATURES_SPEC}-${FEATURES_LANG}-${NORMFLAG}.data

# LSTMs in the encoder are bi-directional, thus the decoder size must be twice the size of the encoder
encoder_size=$((256))
decoder_size=$((${encoder_size}*2))
ENC_FFN_DIM=$((${decoder_size}*2))
NCONV=1
NLSTM=3
ATT_HEADS=2
ELAYERS=1
DLAYERS=3
DROP_RATIO=0.4
TRANS_DROP_RATIO=0.12
MAX_EPOCHS=150
LR=0.0005
LR_SHRINK=0.98
START_ANNEAL=1
#if [[ ${START_ANNEAL} -eq 1 ]]; then
#	LR=`echo ${LR} ${LR_SHRINK} | awk '{print $1/$2}'`
#fi
WDECAY=0.0001
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
if [[ ${FEATURES_TYPE} == 'FlowBERTS' ]]; then
	NUM_FEATURES=2665
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
DECODER='basic'
CRITERION='slu_ctc_loss' # cross-entropy doesn't make sense with the basic decoder.

if [[ $# -ge 1 ]]; then
	LR=$1
fi

wu_epochs=0
wu_updates=$((${wu_epochs}*4627))
warmup_opt="" #"--warmup-updates ${wu_updates}" # with Batch 5: 5393/epoch for MEDIA, 4627/epoch for FSC
SAVE_PATH=TALN2022_${FEATURES_TYPE}-${FEATURES_HEAD}${FEATURES_SPEC}_Dec-${DECODER}-${SUBTASK}-${CORPUS}_BATCH${BATCH}_LR${LR}_NewInit_WU${wu_epochs}epochs_${NLSTM}LSTM-${encoder_size}_Dropout${DROP_RATIO}_StartAnneal${START_ANNEAL}_SCRATCH_TEST/
if [[ $# -ge 2 ]]; then
	SAVE_PATH=${2}_LR${LR}/
fi

reg_options="--clip-norm ${CLIP_NORM} --weight-decay ${WDECAY}"
#CUDA_VISIBLE_DEVICES=1
PYTHONPATH=${HOME}/work/tools/fairseq/ ${FAIRSEQ_PATH}/fairseq-train ${DATA_PATH} --corpus-name ${CORPUS} --feature-extension ${FEATURES_EXTN} \
	--task end2end_slu --arch end2end_slu_arch --criterion ${CRITERION} --num-workers=0 --distributed-world-size 1 \
	--decoder ${DECODER} --padded-reference \
	--save-dir ${SAVE_PATH} --patience 20 --no-epoch-checkpoints \
	--speech-conv ${NCONV} --num-features ${NUM_FEATURES} --speech-conv-size ${encoder_size} --drop-ratio ${DROP_RATIO} \
	--num-lstm-layers ${NLSTM} --speech-lstm-size ${encoder_size} --window-time ${WINTIME} --w2v-language ${FEATURES_LANG} \
	--encoder-normalize-before --encoder-layers ${ELAYERS} --encoder-attention-heads ${ATT_HEADS} --encoder-ffn-embed-dim ${ENC_FFN_DIM} \
	--decoder-normalize-before --attention-dropout ${TRANS_DROP_RATIO} --activation-dropout ${TRANS_DROP_RATIO} \
	--decoder-layers ${DLAYERS} --share-decoder-input-output-embed \
	--encoder-embed-dim ${encoder_size} --encoder-hidden-dim ${encoder_size} --decoder-embed-dim ${encoder_size} --decoder-hidden-dim ${encoder_size} \
	--dropout ${TRANS_DROP_RATIO} --decoder-attention-heads ${ATT_HEADS} \
	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} \
	--lr-scheduler fixed --force-anneal ${START_ANNEAL} --lr-shrink ${LR_SHRINK} \
	--optimizer adam --lr ${LR} ${reg_options} ${warmup_opt} \
	--max-sentences ${BATCH} --max-epoch ${MAX_EPOCHS} --curriculum 1 --keep-best-checkpoints 5

#deactivate
#conda deactivate


