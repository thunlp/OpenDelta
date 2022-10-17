
PATHBASE=/mnt/sfs_turbo/hsd/officialod/OpenDelta-1/examples/examples_prompt/
PYTHONPATH=/mnt/sfs_turbo/zhangshudan/anaconda3/envs/officialod/bin/python
PLMPATHBASE=/mnt/sfs_turbo/hsd/plm_cache/ # must be empty string or dir that ends with /
DATASETSPATHBASE=/mnt/sfs_turbo/hsd/huggingface_datasets/saved_to_disk/
RUNTIME=$(date +%m%d%H%M%S)
MODELNAME="roberta-base"
DATASET=$1
DELTATYPES=("adapter")
CUDAIDS=$2
NUMTRIALS=50
CONTINUESTUDY=${3:-'0'}

echo $RUNTIME
echo $MODELNAME
echo $DATASET
echo $DELTATYPE
echo $CUDAIDS
echo $NUMTRIALS
echo $CONTINUESTUDY
cd $PATHBASE



for expid in 0
do
    ( $PYTHONPATH search_distributed.py \
    --model_name $MODELNAME \
    --dataset $DATASET \
    --delta_type ${DELTATYPES[$expid]} \
    --cuda_ids ${CUDAIDS[$expid]} \
    --num_trials $NUMTRIALS \
    --mode run \
    --repeat_time 1 \
    --main_file_name run_mlm.py \
    --pathbase $PATHBASE \
    --pythonpath $PYTHONPATH \
    --plm_path_base $PLMPATHBASE \
    --datasets_saved_path $DATASETSPATHBASE \
    --datasets_load_from_disk \
    --continue_study $CONTINUESTUDY >>/mnt/sfs_turbo/hsd/officialod/OpenDelta-1/examples/examples_prompt/out_sfs/$RUNTIME.txt 2>&1
    ) &
done
wait