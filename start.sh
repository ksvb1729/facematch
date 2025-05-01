#!/bin/bash  
apt-get update  
apt-get install -y build-essential cmake gfortran libopenblas-dev liblapack-dev  
pip install --upgrade pip  
pip install -r requirements.txt 
