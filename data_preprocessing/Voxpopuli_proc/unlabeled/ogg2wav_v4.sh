#!/bin/sh

list="2012 2013 2014 2015 2016 2017 2018 2019 2020 2009 2010 2011"
set -- $list

(for file in $7/*.ogg; do filename=${file#$7/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$7/16000/${filename%.*}.wav" ; done) & (for file in $8/*.ogg; do filename=${file#$8/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$8/16000/${filename%.*}.wav" ; done) & (for file in $9/*.ogg; do filename=${file#$9/} ; sox -S $file -r 16000 -c 1 -b 16 -e signed-integer "$9/16000/${filename%.*}.wav" ; done)
