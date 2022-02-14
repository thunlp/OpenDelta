<div align="center">


<img src="https://s4.ax1x.com/2022/02/14/Hy7lAf.png" width="350px">

**An Open-Source Framework for Paramter Efficient Tuning.**

------

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="https://opendelta.readthedocs.io/en/latest/notes/usage.html">Basic Usage</a> • 
  <a href="https://opendelta.readthedocs.io/">Docs</a> • 
  <a href="https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing">Performance</a> •


</p>

</div>

![version](https://img.shields.io/badge/version-0.0.1-blue)

## Overview

OpenDelta is a toolkit for parameter efficient methods (we dub it as *delta tuning*), by which users could flexibly assign (or add) a small amount parameters to update while keeping the most paramters frozen. By using OpenDelta, users could easily implement prefix-tuning, adapters, Lora, or any other types of delta tuning with preferred PTMs.

Our repo is tested on Python 3.8 and PyTorch 1.9.0. Lower version may also be supported. 

## Installation
create a virtualenv (optional)
```shell
conda create -n opendelta_env python=3.8
conda activate opendelta_env
```

### Using Pip



Install OpenDelta using pip as follows:
```shell
pip install opendelta
```

To play with the latest features, you can also install OpenDelta from the source.

### Build from Source

```shell
git clone https://github.com/thunlp/OpenDelta.git
cd OpenDelta
``` 

#### Option 1: If you won't modify the code, run
```shell
python setup.py install
```

#### Option 2:  If you want to modify the code, run
```shell
python setup.py develop
```



### Verified Supported Models

**You can try to use OpenDelta on *any* backbone models based on PyTorch.**  
However, with small chances that
The interface of the submodules of the backbone model is not supported. Therefore we verified some commonly
used models that OpenDelta are sure to support.

We will keep testing more and more emerging models.

Pull requests are welcomed when you successfully apply OpenDelta on your own backbone model.


|            | Lora | Bias<br>Tuning  | Adapter<br>Houstbly | Adapter<br>Preffier  | Adapter<br>Drop  | Adapater<br> Low-Rank   | Compactor  |Prefix<br> Tuning      | Prompt <br> Tuning |
| --------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ----- | 
| T5             | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
| GPT-2          | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     |
| BART           | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     | 
| DistilBERT     | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     | 
| RoBERTa        | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     |
| BERT           | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
| T5-3b(parallel)| ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |
| Deberta-v2     | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     |     |
| CTRL           | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  | ✅  |     |     |
| ViT            | ✅  |     |    |     |     |      |   |     |     |


### Performance Checked Combination

Google sheet [here](https://docs.google.com/spreadsheets/d/1BIVa8ocAPga-u7rBOXLYaTfaJSjI1dWfwohmLjmFDrY/edit?usp=sharing)

Subject to change at any moment. 

