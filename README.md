git clone https://github.com/zijianan/activeLearning_with_Fastfit.git  
conda create --name test_fu python=3.9  
conda activate test_fu  
pip install -r requirements.txt --timeout 100  
python -m ipykernel install --user --name test_fu --display-name "test_fu" 
 
CCR (need fill in "gcc/11.2.0 openmpi/4.1.1 pytorch/1.13.1-CUDA-11.8.0" in the job require page)  
module load gcc/11.2.0  
module load openmpi/4.1.1  
module load pytorch/1.13.1-CUDA-11.8.0  
python -m venv /projects/academic/kjoseph/zijian/test_fu  
source /projects/academic/kjoseph/zijian/test_fu/bin/activate  
cd activeLearning_with_Fastfit  
pip install -r requirements.txt --timeout 100  
python -m ipykernel install --user --name test_fu --display-name "test_fu"  
we can also run a teminal in ccr "ssh -NL 3306:localhost:3306 zijianan@hasek.cse.buffalo.edu" to listen mysql data from hasek into ccr  
