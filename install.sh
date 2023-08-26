export NCCL_DEBUG=""
python -m pip install -e transformers/
python -m pip install accelerate
python -m pip install deepspeed==0.10.0
python -m pip install torch torchvision torchaudio
python -m pip install nltk
python -m pip install rouge-score
python -m pip install torchtyping
python -m pip install datasets
python -m pip install openai
python -m pip install backoff