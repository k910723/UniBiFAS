# nohup bash /home/kevin/UniBiFAS/job_random_config.sh &
# bash /home/kevin/UniBiFAS/job_random_config.sh

for i in {1..10}; do
    #python main.py --config configs/unibifas_random.yaml --op_dir ./output --source CIM --target O --randomize
    python main.py --config configs/unibifas_random.yaml --op_dir ./output --source OIM --target C --randomize
    #python main.py --config configs/unibifas_random.yaml --op_dir ./output --source OCM --target I --randomize
    #python main.py --config configs/unibifas_random.yaml --op_dir ./output --source OCI --target M --randomize
done