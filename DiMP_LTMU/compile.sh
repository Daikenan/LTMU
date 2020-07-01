#!/bin/bash


echo "****************** Siammask Network ******************"
cd SiamMask/sm_utils/pyvotkit
python setup.py build_ext --inplace
cd ../../../

cd SiamMask/sm_utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../../


echo "****************** Global_Track Network ******************"
cd Global_Track/_submodules/mmdetection
python setup.py develop
cd ../../../





