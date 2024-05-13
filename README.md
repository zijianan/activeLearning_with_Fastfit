conda create --name test_fu python=3.9  
conda activate test_fu  
pip install -r requirements.txt --timeout 100  
python -m ipykernel install --user --name test_fu --display-name "test_fu" 
 
CCR:module load gcc/11.2.0  
    module load openmpi/4.1.1  
    module load pytorch/1.13.1-CUDA-11.8.0  
