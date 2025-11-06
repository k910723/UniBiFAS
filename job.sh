# nohup bash /home/kevin/UniBiFAS/job.sh &

python main.py --config configs/unibifas.yaml --op_dir ./ablation_output --source OCM --target I --seed 1
python main.py --config configs/unibifas_TI.yaml --op_dir ./ablation_output --source OCM --target I --seed 1

python main.py --config configs/unibifas_bi_IT.yaml --op_dir ./ablation_output --source OCM --target I --seed 1
python main.py --config configs/unibifas_bi_TI.yaml --op_dir ./ablation_output --source OCM --target I --seed 1

python main.py --config configs/unibifas_uni_IT.yaml --op_dir ./ablation_output --source OCM --target I --seed 1
python main.py --config configs/unibifas_uni_TI.yaml --op_dir ./ablation_output --source OCM --target I --seed 1