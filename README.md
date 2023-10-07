# MLOps
Project for MLOps course

## How to run

```
git clone https://github.com/ShieldVP/MLOps.git
cd MLOps
python -m venv venv
source venv/bin/activate
poetry install
pre-commit install
pre-commit run -a
python MLOps/train.py
python MLOps/infer.py
```
