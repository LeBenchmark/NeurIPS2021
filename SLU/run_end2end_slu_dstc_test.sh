#!/bin/bash

#CUDA_VAR="CUDA_VISIBLE_DEVICES=1"
# TO RUN ON DECORE AND OTHER SERVERS UNCOMMENT THE FOLLOWING LINES
#export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
#export PYTHONPATH=${HOME}/anaconda3/

# TO RUN ON THE LIG GRID WITH OAR UNCOMMENT THE FOLLOWING LINES
source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/

#export RNNTAGGERPATH=${HOME}/work/tools/Seq2Biseq_End2EndSLU/

GENERATE=1
EVALUATE=1
echo " ---"
echo " * Using python: `which python`"
echo " * Using fairseq-train: `which fairseq-train`"; echo
if [[ ${GENERATE} -eq 0 ]]; then
	echo " *** No generation, evaluating model only..."
fi

# -----------------
ENSEMBLE_SCRIPT=${HOME}/work/tools/fairseq/scripts/average_checkpoints.py
FEATURES_TYPE='FlowBERT' # Use 'spectro' or 'W2V'
FEATURES_LANG='Fr'      # Use 'Fr' or 'En' ('En' only with 'W2V')
NORMFLAG='Normalized'
# -----------------
DATA_PATH=${HOME}/work/data/MEDIA-Original/semantic_speech_aligned_corpus/DialogMachineAndUser_SemTextAndWav_FixedChannel/
SERIALIZED_CORPUS=MEDIA.user+machine.${FEATURES_TYPE}-7Klt-${FEATURES_LANG}-${NORMFLAG}.data
SUBTASK=concept
if [[ $# -eq 3 ]]; then
	SUBTASK=$3
fi

CRITERION='cross_entropy' # Use 'slu_ctc_loss' for CTC loss, 'cross_entropy' for Cross Entropy loss

CHECKPOINT=checkpoints/checkpoint_best.pt
if [[ $# -ge 1 ]]; then
	CHECKPOINT=$1
fi
if [[ $# -ge 2 ]]; then
	SERIALIZED_CORPUS=$2
	#if [[ `ls -l ${SERIALIZED_CORPUS}.* | wc -l` -eq 0 ]]; then
	#	echo "$0 ERROR: no serialized data found with prefix ${SERIALIZED_CORPUS}"; exit 1
	#fi
fi

echo " * Using serialized data prefix: ${SERIALIZED_CORPUS}"
echo " ---"

beam_size=1
BASIC_OPTIONS="--beam 1 --iter-decode-max-iter 1 --prefix-size 0 --match-source-len"
GENERATE_OPTIONS="--beam ${beam_size} --iter-decode-max-iter ${beam_size} --max-len-a 1.0 --max-len-b 25 --prefix-size 0"	# Average max-len-a for MEDIA (computed on train): 0.123

ctx_fusion='sum'
ROBERTA_MODEL='/home/getalp/dinarelm/work/data/models/Roberta/roberta.large/model.pt'
AUX_OPTIONS="--corpus-name fsc --use-transcription-as-input --dialog-level-slu --normalize-dialog-batches --use-dialog-history --context-fusion ${ctx_fusion} --context-size 4 --context-first-turns 0 --context-discount 0.5" #--roberta-model ${ROBERTA_MODEL}"

GPU_ID=1
BATCH=1
CHECKPOINT_DIR=`dirname ${CHECKPOINT}`
if [[ ${GENERATE} -ne 0 ]]; then
	ENSEMBLE_MODEL=${CHECKPOINT_DIR}/checkpoint.ensemble.pt
	if [[ ! -e ${ENSEMBLE_MODEL} ]]; then
		echo " * Generating ensemble model..."

		python ${ENSEMBLE_SCRIPT} --inputs ${CHECKPOINT_DIR}/checkpoint.best_loss* --output ${ENSEMBLE_MODEL}
		echo "Done."
	fi

	#CUDA_VISIBLE_DEVICES=${GPU_ID}
	${FAIRSEQ_PATH}/fairseq-generate ${DATA_PATH} ${AUX_OPTIONS} \
		--path ${ENSEMBLE_MODEL} ${GENERATE_OPTIONS} --max-sentences ${BATCH} --num-workers=0 \
		--task end2end_slu --criterion ${CRITERION} --padded-reference --model-beam ${beam_size} \
		--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} --user-only \
		--gen-subset valid --results-path ${CHECKPOINT_DIR} --sacrebleu

	#CUDA_VISIBLE_DEVICES=${GPU_ID}
	${FAIRSEQ_PATH}/fairseq-generate ${DATA_PATH} ${AUX_OPTIONS} \
        	--path ${ENSEMBLE_MODEL} ${GENERATE_OPTIONS} --max-sentences ${BATCH} --num-workers=0 \
        	--task end2end_slu --criterion ${CRITERION} --padded-reference --model-beam ${beam_size} \
        	--serialized-data ${SERIALIZED_CORPUS} --slu-subtask ${SUBTASK} --user-only \
        	--gen-subset test --results-path ${CHECKPOINT_DIR} --sacrebleu
fi

SCORING_SCRIPT='/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/scripts/score_model_output.sh'

valid_name=${CHECKPOINT_DIR}/generate-valid.txt
test_name=${CHECKPOINT_DIR}/generate-test.txt
if [[ ${EVALUATE} -eq 1 ]]; then
	if [[ -f ${valid_name} ]]; then
		echo " ### evaluating on dev set (raw output):"
		${SCORING_SCRIPT} ${valid_name} '--clean-hyp' | tee ${CHECKPOINT_DIR}/generate-valid.raw-eval
		echo " ##### evaluating on dev set (slu output):"
		${SCORING_SCRIPT} ${valid_name} '--clean-hyp --slu-out' | tee ${CHECKPOINT_DIR}/generate-valid.slu-eval
		echo " *** -------------------- ***"
	fi
	if [[ -f ${test_name} ]]; then
		echo " ### evaluating on test set (raw output):"
		${SCORING_SCRIPT} ${test_name} '--clean-hyp' | tee ${CHECKPOINT_DIR}/generate-test.raw-eval
		echo " ##### evaluating on test set (slu output):"
		${SCORING_SCRIPT} ${test_name} '--clean-hyp --slu-out' | tee ${CHECKPOINT_DIR}/generate-test.slu-eval
		echo " *** -------------------- ***"
	fi

	echo " ### RESULTS SUMMARY @${CHECKPOINT_DIR}:"
	if [[ -f ${valid_name} ]]; then
		echo " - Dev set raw error rate:"
		grep "Total error rate:" ${CHECKPOINT_DIR}/generate-valid.raw-eval
		echo " - Dev set slu error rate:"
		grep "Total error rate:" ${CHECKPOINT_DIR}/generate-valid.slu-eval
	fi
	if [[ -f ${test_name} ]]; then
		echo " - Test set raw error rate:"
		grep "Total error rate:" ${CHECKPOINT_DIR}/generate-test.raw-eval
		echo " - Test set slu error rate:"
		grep "Total error rate:" ${CHECKPOINT_DIR}/generate-test.slu-eval
	fi
fi

deactivate
#conda deactivate



