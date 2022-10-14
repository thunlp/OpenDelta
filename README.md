<div align="center">


<img src="https://s4.ax1x.com/2022/02/14/Hy7lAf.png" width="350px">

**An Open-Source Framework for Paramter-Efficient Tuning (Delta Tuning).**

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

OpenDelta is a toolkit for parameter-efficient tuning methods (we dub it as *delta tuning*), by which users could flexibly assign (or add) a small amount parameters to update while keeping the most paramters frozen. By using OpenDelta, users could easily implement prefix-tuning, adapters, Lora, or any other types of delta tuning with preferred PTMs.

- Our repo is tested on Python 3.=-0 and PyTorch 1.9.0. Lower version may also be supported. 

- **A demo of using Opendelta to modify the PLM (E.g., BART).**
![How PLM changes using Delta-tuning](docs/source/imgs/demo.gif)

## News
- **2022.10.14** Release v0.3.0. We make the usage of default configurations of each delta tuning methods (i.e., the position they are attached) more friendly! If a custom model has our supported models as submodules inside, the default configuration is also available. Other key changes can be seen in [Update Log](file:///Users/hsd/codes/opendelta_doc/OpenDelta/docs/build/html/notes/update.html#version-0-3-0)
- **2022.10.10** Merge a long-developed branch v0.2.4 into the master branch. Key updates are (1) the an example unifying the delta tuning paradigm and the prompt-tuning paradigm; (2) and support for [delta centers](https://www.openbmb.org/toolKits/deltacenter), whose webpage is still under construction. Details can be seen in [Update Log](file:///Users/hsd/codes/opendelta_doc/OpenDelta/docs/build/html/notes/update.html#version-0-2-4)
- **2022.03.24** We notice several bugs in Soft Prompt Tuning and Prefix Tuning, mainly due to their need to customize attention ids, token_type_ids, we are fixing it! Currently, please use the other methods since they are stabler and better in performance. 
- **2022.03.20** Add a [colab example](https://colab.research.google.com/drive/1uAhgAdc8Qr42UKYDlgUv0f7W1-gAFwGo?usp=sharing) to illustrate efficient training and space-saving multitask-serving.
- **2022.03.20** A new pip version released.
- **2022.02.16** Support [regular expression](https://opendelta.readthedocs.io/en/latest/notes/namebasedaddr.html#regexexpr) in named-based addressing. 

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

#### Option 2:  If you want to modify the code or keep the repo updated by git clone, run
```shell
python setup.py develop
```

#### Tips
- If you want to use mirror for installing the packages, please change the `index_url` in [setup.cfg](setup.cfg)

- If you encounter network error using setup.py, please firstly install the dependencies via
```shell
pip install -r requirements.txt && python setup.py develop
```

## Must Try

```python
from transformers import AutoModelForSeq2SeqLM
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
from opendelta import AutoDeltaModel
delta = AutoDeltaModel.from_finetuned("thunlp/FactQA_T5-large_Adapter", backbone_model=t5)
delta.log()
```

## Verified Supported Models

- **You can try to use OpenDelta on *any* backbone models based on PyTorch.**  
- However, with small chances thatThe interface of the submodules of the backbone model is not supported. Therefore we verified some commonly
used models that OpenDelta are sure to support.

- We will keep testing more and more emerging models.

- Pull requests are welcomed when you successfully apply OpenDelta on your own backbone model.


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





