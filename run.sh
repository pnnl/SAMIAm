#!/bin/bash
export PATH=/share/apps/python/miniconda4.12/bin:/share/apps/gcc/9.1.0/bin:/people/scicons/deception/bin:/qfs/people/abeb563/.vscode-server/bin/b3e4e68a0bc097f0ae7907b217c1119af9e03435/bin/remote-cli:/people/scicons/deception/bin:/share/apps/python/miniconda4.12/condabin:/usr/lib64/qt-3.3/bin:/people/scicons/deception/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ganglia/bin:/opt/ganglia/sbin:/opt/pdsh/bin:/opt/rocks/bin:/opt/rocks/sbin:/people/abeb563/bin:/people/abeb563/bin:/people/abeb563/.local/bin

#module load python/miniconda4.12
module load python/miniconda3.8
module load cuda/11.1

#source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh
pip install matplotlib
pip install opencv-python
pip install scikit-learn
pip install torch==1.7.1 #--upgrade
pip install torchvision==0.8.2 #--upgrade
pip install torch-encoding #--upgrade
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pandas

#python supervised.py  --model=res18
#python unsupervised.py --model=FENet --embed=0 --post=1

python batch_unsupervised.py --model=FENet --embed=0 --post=1

#python preprocess.py