# !!!!This example collection is still under develop, please wait for some time to use it.

## install the repo
```bash
cd ../
python setup_seq2seq.py develop
```
This will add `examples_seq2seq` to the environment path of the python lib.

## Generating the json configuration file

```shell
python configs/gen_$BACKBONETYPE.py --job $YOURJOB
#e.g. python configs/gen_beit.py --job lora_beit-base-patch16-224
```
The available job configuration (e.g., `--job lora_beit-base-patch16-224`) can be seen from the scripts. You can also
create your only configuration.


## Run the code

```
CUDA_VISIBLE_DEVICES=1 python src/run.py configs/lora_beit-base-patch16-224/beans.json
```
