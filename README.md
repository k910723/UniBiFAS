python main.py --config configs/unibifas.yaml --op_dir ./output --source OCI --target M

ToDo:
Current implementation does not calculate the hierarchichical loss using the class tokens of early layers.
See engine/trainer.py

Current source dataset and target dataset have to be specified in argument instead of in the config file.

check SCM normalization.

now the segmentation loss only calculate similarity between spoof text and patches tokens.

beware the threhold calculation