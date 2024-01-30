module load python3/3.10.0 && source venv/bin/activate && export PYTHONPATH=`realpath ../.local/lib/`

python3 -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://`hostname`:40000 --model-path ./checkpoints/astrollava-v1.5-7b-lora --model-base lmsys/vicuna-7B-v1.5
