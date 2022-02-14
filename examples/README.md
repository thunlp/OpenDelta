# Use Examples

This repo mainly contains several running scripts to use OpenDelta to conduct parameter-efficient training of various tasks.

**Note that we suggest adding OpenDelta to existing scripts, instead of modify a scripts into the following examples. OpenDelta itself doens't restrict the training pipeline nor provide pipeline.**


## tutorial
Several toy tutorials:
1. The scripts for docs/basic_usage
2. Using interactive module selection
3. Work with [OpenPrompt](https://github.com/thunlp/OpenPrompt)

## examples_text-classification
Modify a huggingface text-classification examples into a delta tuning one.
Currently, GLUE datasets are supported in the scripts. Roberta-base is used for performance checking. Read README.md inside the repo for detailed usage.

## examples_seq2seq
Modify a huggingface sequence to sequence examples into a delta tuning one.
Currently, SuperGLUE and GLUE datasets are supported in the scripts. T5-base is used for performance checking. Read README.md inside the repo for detailed usage.


## examples_image-classification
A toy example of using OpenDelta for a Computer Vision Pretrained Model (ViT). Since ViT is an experimental feature in huggingface transformers, this example is subject to Change at any moment. 

