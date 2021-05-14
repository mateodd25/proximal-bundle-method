#!/bin/bash
# This script is based on
# https://github.com/google-research/FirstOrderLp.jl/blob/md/L1SVM/benchmarking/collect_LIBSVM.sh.

# Collects the three datasets from LIBSVM used in the experiments.
# It assumes that the environment has curl and bunzip2 installed
# and working.

local_data_directory=./data
mkdir $local_data_directory

# To download the binary problems:
data_source="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
for filename in colon-cancer duke leu;
do
    curl $data_source/$filename.bz2 --output $local_data_directory/$filename.bz2
    bunzip2 -d $local_data_directory/$filename.bz2
done
