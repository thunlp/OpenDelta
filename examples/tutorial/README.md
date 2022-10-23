
## 0_basic.py
The scripts for docs/basic_usage

## 1_interactive
Using interactive module selection
We suggest run it in vscode terminal where port mapping is automatic done for you.

## 1_with_openprompt.py
Integrate opendelta with openprompt.

requirement: 
```
pip install openprompt

```

## 2_with_bmtrain.py
1. install necessary packages:
```
pip install git+https://github.com/OpenBMB/BMTrain.git
pip install git+git@github.com:OpenBMB/ModelCenter.git
```
2. download dataset from https://super.gluebenchmark.com/tasks, e.g., 
```
mkdir down_data
mkdir down_data/superglue
cd down_data/superglue
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip
unzip BoolQ.zip
cd ../../
```
3. Run the shell scripts, change `NNODES`,`GPUS_PER_NODE` according to your computational resources.
```
bash 2_with_bmtrain.sh 
```

