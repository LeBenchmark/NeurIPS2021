#!/bin/sh


#list="$1 $2 $3"
#set -- $list
mkdir $1/16000
mkdir $2/16000
mkdir $3/16000

(for file in $1/*.ogg; do filename=${file#$1/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$1/16000/${filename%.*}.wav" ; done) & (for file in $2/*.ogg; do filename=${file#$2/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$2/16000/${filename%.*}.wav" ; done) & (for file in $3/*.ogg; do filename=${file#$3/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$3/16000/${filename%.*}.wav" ; done)
