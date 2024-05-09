conda create --name test_fu python=3.9 \n
conda activate test_fu \n
pip install -r requirements.txt --timeout 100 \n
python -m ipykernel install --user --name test_fu --display-name "test_fu"
