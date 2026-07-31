## setup:------------>commanda used are below 

git --version

python -m venv .venv 

.venv\Scripts\activate

python -m pip install --upgrade pip

nvidia-smi

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

 python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

pip install the core libraries..present in requirement.txt detail name

pip freeze > requirements.txt

 "" > train.py; "" > evaluate.py; "" > infer.py; "" > README.md; "" > .gitignore

 python -c "import torch, cv2, timm, pywt, albumentations, fastapi; print('Environment Ready')"

## module:-1
for testing
pip install pytest

## module :-2

## module 3:- security geetkeeper..

 Get-Content .\backbone\grn.py
 python -c "import backbone.grn; print(backbone.grn.__file__)"

 ## for formatting code correctly,,,looking nice some commands:-->
 black backbone tests
isort backbone tests
ruff check backbone tests
pytest -v
Format-Hex attention\attention_models.py | Select-Object -First 5
## delete caches
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .ruff_cache -ErrorAction SilentlyContinue
python -m py_compile representation/base_encoder.py

python -m compileall .

tree /F  -->full folder st
dir representation
dir representation\*.py
python -m compileall cross_representation
python -m pytest

python -c "from pathlib import Path; data=Path('ufre/base_reasoner.py').read_bytes(); print('NULL bytes:', b'\x00' in data)"

python -c "from pathlib import Path; import os; [print(f, b'\x00' in Path(f).read_bytes()) for f in Path('ufre').glob('*.py')]"

pip install pytest-cov
pytest --cov=. --cov-report=term-missing
pytest --cov=. --cov-report=html
python -m tests.test_pipeline

findstr /S /N /C:"StageRepresentation(" *.py