export NCCL_DEBUG=""
python -m pip install transformers==4.21.1
python -m pip install deepspeed==0.7.3
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install nltk
python -m pip install rouge-score
python -m pip install torchtyping
python -m pip install datasets
python -m pip install openai
python -m pip install backoff