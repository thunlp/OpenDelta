(saveload)=
# Save and Share the Delta

## Space efficient saving without changing the code.
After a modified backbone model is trained, you can save only trained part without change to any code, because **the state dict of the backbone model has been changed to the trainable parts**

```python
from opendelta import CompacterModel
from transformers import BertForMaskedLM
backbone_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
delta_model = CompacterModel(backbone_model) # modify the default modules.

# freeze module
delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
# or
delta_model.freeze_module(exclude=["deltas"])
```
### save the checkpoint.
now save the backbone_model in normal way, and the checkpoint is **very space efficient**.

```python
# ...
# After some training pipeline
# ...
torch.save(backbone_model.state_dict(), "delta.ckpt")

# the checkpoint size
import os
print("checkpoint size: {:.2f}M".format(os.path.getsize("delta.ckpt")/1024**2))
# checkpoint size: 0.32M
```

### load the checkpoint.
In order to load the checkpoint, you should make sure the backbone model is a modified ones (so that it can take in the delta parameters).
Then load the checkpoint with `strict=False`.
```python
backbone_model.load_state_dict(torch.load("delta.ckpt"), strict=False)
# this will return long string of warning about the 'missing key'.
# if you want to supress it, use
# _ = backbone_model.load_state_dict(torch.load("delta.ckpt"), strict=False)
```

## Save/Load the entire model after training.

### save a delta model.
```python
delta_model.save_finetuned("delta_model")
# Configuration saved in delta_model/config.json
# Model weights saved in delta_model/pytorch_model.bin
```
This will save all the trained parameters and the configuration of the delta model to path `delta_model/`

### load a delta model.

```python
backbone_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
delta_model.from_finetuned("delta_model", backbone_model, local_files_only=True) 
# passing local_files_only=True will save the time of checking in the web.
```

## Share or download a model to/from the community.

### Share.
```python
delta_model.save_finetuned("test_delta_model", push_to_hub = True)
```

###  Download from community.
```python
from transformers import AutoModelForSeq2SeqLM
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
from opendelta import AutoDeltaModel
delta = AutoDeltaModel.from_finetuned("DeltaHub/lora_t5-base_mrpc", backbone_model=t5)
delta.log()
```

<div class="admonition tip"> 
<p class="title">**Push to Hub**</p>
<p> Currently we only provide the option to push to huggingface model hub.</p>
<p> Before push to hub, you may need to register an account on Huggingface. You can refer to this [tutorial about model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing)
</p>
<p> In some cases, your checkpoint is still large for git, please install [`git-lfs`](https://git-lfs.github.com).
</p>
</div>

:::{admonition} **Sharing with the Community**
:class: tip
If you are satisfied with your checkpoint, do not forget to share your model to <a href="https://huggingface.co/DeltaHub">DeltaHub</a>:
1. Add yourself to DeltaHub with the [public link](https://huggingface.co/organizations/DeltaHub/share/QzkBuLSmlVnNhQqHYnekoTXwSRkoRHBwZA)
2. Be sure to edit your model card to clearly illustrate the delta model before you share.
3. Click `setting` on the model
4. Transfer the model in `rename or transfer this model` section.
::: 


## Save & Load for Composition of Delta

<img src="../imgs/todo-icon.jpeg" height="30px"> Currently save & load method is not suitable for [composition of delta model](compositon). Please wait for future releases. 