#!/bin/bash

source ${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/activate 
export FAIRSEQ_PATH=${HOME}/work/tools/venv_python3.7.2_torch1.4_decore0/bin/
export PYTHONPATH=${PYTHONPATH}:${HOME}/work/tools/fairseq/
script_path=/home/getalp/dinarelm/work/tools/fairseq_tools/end2end_slu/scripts/

model_output=$1
aux_opt=""
if [[ $# -ge 2 ]]; then
	aux_opt=$2
fi

#grep "^T\-" ${model_output} | cut -f 2- | perl -pe '{s/   / \# /g;}' > ${model_output}.ref # "s/   / \# \g;" is to keep into account spaces
#grep "^H\-" ${model_output} | cut -f 3- | perl -pe '{s/   / \# /g;}' > ${model_output}.hyp
#grep "^P\-" ${model_output} | cut -f 2- > ${model_output}.scores
grep "^T\-" ${model_output} | perl -pe '{s/   / \# /g;}' > ${model_output}.ref # "s/   / \# \g;" is to keep into account spaces
grep "^H\-" ${model_output} | perl -pe '{s/   / \# /g;}' > ${model_output}.hyp
grep "^P\-" ${model_output} > ${model_output}.scores

# --clean-hyp removes blanks and duplicate tokens generated when training with CTC loss.
# Remove this option if you trained the model with another loss (e.g. cross entropy)
# --slu-out keeps only concepts from the raw output. Use this option if you want to score the model with Concept Error Rate (CER)
python ${script_path}/compute_error_rate.py ${aux_opt} --ref ${model_output}.ref --hyp ${model_output}.hyp --slu-scores ${model_output}.scores

deactivate

