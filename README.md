conda create --name test_fu python=3.9  
conda activate test_fu  
pip install -r requirements.txt --timeout 100  
python -m ipykernel install --user --name test_fu --display-name "test_fu"
