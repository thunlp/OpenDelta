<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Use OpenDelta in vision transformer ViT

This example uses the [huggingface image classification examples](), by adding several
lines in the original scripts.

## Usage
### 1. install necessary package
```shell
pip install Pillow
pip install torchvision
pip install transformers==4.16.2
pip install datsets==1.18.0
```

### 2. run
```bash
python run_image_classification.py configs/lora_beans.json
```

Do not forget to re-install datasets back into 1.17.0 for other examples. :)


## Possible Errors
1. dataset connection error

Solution 1: open a python console, running the error command again, may not be useful

Solution 2: download the dataset by yourself on a internect connected machine, saved to disk and transfer to your server, at last load_from_disk.


## Link to original training scripts
You may find solution to other question about the scripts and irrelevant to Opendelta in 
https://github.com/huggingface/transformers/tree/master/examples/pytorch/image-classification

