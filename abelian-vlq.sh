#!/bin/bash

int='^[0-9]+$'

if [ "$#" -ne 2 ] || ! [[ "$1" =~ $int ]] || ! [[ "$2" =~ $int ]]
then
    echo Error: wrong input   
    echo Usage: abelian-vlq n_up_vlq n_down_vlq
else
    if ! [ -d ./output ]
    then
        mkdir ./output
    fi
    python3 src/texture_zeros.py $1 $2
    python3 src/minimisation.py output/$1_up_$2_down_VLQ_MRT_before_minimisation.dat
    python3 src/abelian_symmetry_2HDM.py output/$1_up_$2_down_VLQ_MRT_after_minimisation.dat
    python3 src/notation.py output/$1_up_$2_down_VLQ_MRT_after_minimisation.dat output/$1_up_$2_down_VLQ_MRT_after_symmetry.dat
fi
