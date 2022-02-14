(autodelta)=
# AutoDelta Mechanism

Inspired by [Huggingface transformers AutoClasses](https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/auto#transformers.AutoModel) , we provide an AutoDelta features for the users to

1. Easily to experiment with different delta models
2. Fast deploy from configuration file, especially from the repos in [DeltaHub](https://huggingface.co/DeltaHub).


## Easily load from dict, so that subject to change the type of delta models.

```python
from opendelta import AutoDeltaConfig, AutoDeltaModel
from transformers import T5ForConditionalGeneration

backbone_model = T5ForConditionalGeneration.from_pretrained("t5-base")
```

We can load a config from a dict
```python
config_dict = {
    "delta_type":"lora", 
    "modified_modules":[
        "SelfAttention.q", 
        "SelfAttention.v",
        "SelfAttention.o"
    ], 
    "lora_r":4}
delta_config = AutoDeltaConfig.from_dict(config_dict)
```

Then use the config to add a delta model to the backbone model
```python
delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone_model)

# now visualize the modified backbone_model
from opendelta import Visualization
Visualizaiton(backbone_model).structure_graph()
```


````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/t5lora.png
---
width: 600px
name: t5lora
---
```
````



## Fast deploy from a finetuned delta checkpoints from DeltaHub

```python
delta_model = AutoDeltaModel.from_finetuned("DeltaHub/sst2-t5-base", backbone_model=backbone_model)  # TODO: the link may change.
```

<div class="admonition note">
<p class="title">**Hash checking**</p>
Since the delta model only works together with the backbone model.
we will automatically check whether you load the delta model the same way it is trained.
</p>
<p>
We calculate the trained model's [md5](http://some_link) and save it to the config. When finishing loading the delta model, we will re-calculate the md5 to see whether it changes.
<p>Pass `check_hash=False` to disable the hash checking.</p>
</div>