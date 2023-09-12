export NCCL_DEBUG=""
python -m pip install -e transformers/
python -m pip install torch torchvision torchaudio
python -m pip install deepspeed==0.10.0
python -m pip install nltk
python -m pip install numerize
python -m pip install torchtyping
python -m pip install rich
python -m pip install accelerate
python -m pip install datasets