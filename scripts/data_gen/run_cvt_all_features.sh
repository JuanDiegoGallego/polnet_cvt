#!/bin/bash
# filepath: /home/tfg/Desktop/Trabajo_julio/polnet/scripts/data_gen/run_cvt_all_features.sh

for i in {0..6}
do
    echo "Launching cvt_all_features.py with parameter $i"
    python3 /home/tfg/Desktop/Trabajo_julio/polnet/scripts/data_gen/cvt_all_features.py $i
done