# nohup bash /home/kevin/UniBiFAS/job.sh &

# python main.py --config configs/unibifas.yaml --op_dir ./output --source CIM --target O --seed 1

# python main.py --config configs/unibifas_binary_only.yaml --op_dir ./output --source SW --target F --seed $seed

# python main.py --config configs/unibifas_fsw.yaml --op_dir ./output --source SW --target F --seed $seed

for seed in {1..10}; do
    python main.py --config configs/unibifas_binary_only.yaml --op_dir ./output --source SW --target F --seed $seed
    python main.py --config configs/unibifas_binary_only.yaml --op_dir ./output --source FW --target S --seed $seed
    python main.py --config configs/unibifas_binary_only.yaml --op_dir ./output --source FS --target W --seed $seed
done