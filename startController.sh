module load python3/3.10.0 && source venv/bin/activate && export PYTHONPATH=`realpath ../.local/lib/`

python3 -m llava.serve.controller --host 0.0.0.0 --port 10000
