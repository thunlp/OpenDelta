# files=(cola mnli mrpc qnli qqp rte sst2 stsb superglue-boolq superglue-cb superglue-copa superglue-multirc superglue-record superglue-wic superglue-wsc.fixed)
# for ((i=$1; i<=$2; i++))
# do
#     dataset=${files[i]}
#     echo "id$i:$dataset"
#     TOKENIZERS_PARALLELISM=false python run_seq2seq.py configs/$3/$dataset.json
# done 

cd configs

for deltatype in "lora" "adapter"
do
    for modeltype in "t5-base" "t5-large" "t5-3b"
    do
        echo $deltatype
        python config_gen_bs$2.py --job $deltatype\_$modeltype
    done
done

ls
cd ..

for deltatype in "lora" "adapter"
do
    for modeltype in "t5-base" "t5-large" "t5-3b"
        do
             CUDA_VISIBLE_DEVICES=$1 bash run.sh 2 2 $deltatype\_$modeltype\_$2
    done
done




            
