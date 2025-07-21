#!/bin/bash

for i in {0..0}
do 
source .venv_polnet/bin/activate
python3 /home/tfg/Desktop/JuanDiego/PolNet/polnet_curvatubes/scripts/data_gen/all_features_cvt3.py $i
done
