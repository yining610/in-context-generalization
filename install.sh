export NCCL_DEBUG=""
python -m pip install -e transformers/
python -m pip install torch==2.0.1
python -m pip install deepspeed==0.10.0
python -m pip install torchvision==0.15.2
python -m pip install nltk
python -m pip install numerize
python -m pip install rouge-score
python -m pip install torchtyping
python -m pip install rich
python -m pip install accelerate
python -m pip install datasets