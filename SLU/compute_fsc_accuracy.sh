#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 <.hyp.scored file> | file generated after running scripts/score_model_output.sh on generate-valid.txt or generate-test.txt"; exit;
fi
if [[ ! -e $1 ]]; then
	echo "$1: no such file"; exit;
fi

# Combined slots evaluation
grep 'REF:' $1 | cut -d' ' -f2- | perl -pe '{s/_SOC_//g; s/_EOC_//g; s/^ //g; s/ $//g; s/ +/ /g; s/ /\-/g;}' > FSC.ref
grep 'HYP:' $1 | cut -d' ' -f2- | perl -pe '{s/_SOC_//g; s/_EOC_//g; s/^ //g; s/ $//g; s/ +/ /g; s/ /\-/g;}' > FSC.hyp

#Intent only evaluation
#grep 'REF:' $1 | cut -d' ' -f2- | perl -pe '{s/_SOC_//g; s/_EOC_//g; s/^ //g; s/ $//g; s/ +/ /g;}' | awk '{print $1}' > FSC.ref
#grep 'HYP:' $1 | cut -d' ' -f2- | perl -pe '{s/_SOC_//g; s/_EOC_//g; s/^ //g; s/ $//g; s/ +/ /g;}' | awk '{print $1}' > FSC.hyp

nnr=`cat FSC.ref | awk '{print NF}' | sort -u`
nnh=`cat FSC.hyp | awk '{print NF}' | sort -u`
if [[ ${nnr} != "1" || ${nnh} != "1" ]]; then
	echo " FORMAT ERROR in the scoring files FSC.ref or FSC.hyp. Expected only 1 field, found ${nnr} and ${nnh}"; exit;
fi

echo ""
echo " ----------"
paste -d' ' FSC.ref FSC.hyp | awk 'BEGIN{correct=0;} { if($1 == $2){correct=correct+1;}} END{printf("FSC annotation accuracy: %.2f%c\n", (correct/NR)*100, "%")}'
echo " -----"

rm -f FSC.ref FSC.hyp

