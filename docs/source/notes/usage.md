(basics)=
# Basic Usage
Now we introduce the general pipeline to migrate your full-model tuning scripts to a delta tuning one. 

## STEP 1: Load the pretrained models

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base") # suppose we load BART
```

## STEP 2: Add delta modules
We provide two alternatives to add the delta modules.
### 2.1 Modification based on visualization
Suppose we want to make the feedforward layer of each block as our [modification target module](target_module),
We should first know what is the name of the feedforward layer in the BART model by visualization. <img src="../imgs/hint-icon-2.jpg" height="30px"> *For more about visualization, see [Visualization](visualization).*

```python
from opendelta import Visualization
Visualization(model).structure_graph()
```

````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/bart-base.png
---
width: 600px
name: bart-base
---
```
````




We can see from the structure graph that the feed forward layer in Bart is called `model.encoder.layers.$.fc1` and `model.encoder.layers.$.fc2`, where
`$` represent a number from 0-5.  Since we want to apply adapter after *all* the feed forward layers, we specify the `modified_modules=['fc2']`, which is the common suffix for feed forward layers.
<img src="../imgs/hint-icon-2.jpg" height="30px">  *For details about the name based addressing, see [Name-based submodule addressing](namebasedaddr)*

Other configurations, such as the `bottleneck_dim` in Adapter, can be passed as key word arguments.
```python
from opendelta import AdapterModel
delta_model = AdapterModel(backbone_model=model, modified_modules=['fc2'], bottleneck_dim=12)
delta_model.log() # This will visualize the backbone after modification and other information.
```

(target_module)=
:::{admonition} Target module
:class: note
For different delta methods, the operation for the modification target is different.
- Adapter based method: Insert at the target module's forward function.
- BitFit: Add bias to all allowed position of the target module.
- Lora: Substitute the all the linear layers of the target module with [Lora.Linear](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L92).
:::

### 2.2 Use the default modification.
We also provide the default modifications of each delta methods for some commonly used PTMs (e.g., BERT, RoBERTA, DistilBERT, T5, GPT2), so the users don't need to specify the submodules to modify.

The default modifications is achieved by a [common_structure mapping](commonstructure), that is, use the mapping a name of a module to the it's name on a common transformer structure. <img src="../imgs/hint-icon-2.jpg" height="30px">  *For details about the default modification, see [Unified Name Convention](unifyname)*



```python
# a seperate example using BERT.
from transformers import BertForMaskedLM
from opendelta import AdapterModel
model = BertForMaskedLM.from_pretrained("bert-base-cased")
delta_model = AdapterModel(model) # This will apply adapter to the self-attn and feed-forward layer.
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/defaultmodification.png
---
width: 600px
name: defaultmodification
---
```
````




:::{admonition} Delta model vs Backbone model
:class: note
The delta_model **CAN NOT**  be used alone, and its [forward](opendelta.basemodel.DeltaBase.forward) is canceled. The training pipeline should be conducted on the backbone model (In the above example, its the `model`).
:::

:::{admonition} Try different positions
:class: tip
OpenDelta provide the flexibility to add delta to different positions on the backbone model. For example, If you want to move the adapter in the above example after the layer norm of the feed forward layer. The code should be changed into
```python
# continue with the BART example, but not used later.
delta_model = AdapterModel(backbone_model=model, modified_modules=['final_layer_norm'], bottleneck_dim=12)
```
The performance may vary due to positional differences, but there is no academic guarantee that one will outperform the other.
:::


:::{admonition} Favored Configurations
:class: tip
Feel confused about the flexibility that OpenDelta brings? NO WORRY! We will add [Favored Configurations](favoredconfiguration) soon.
:::

## STEP 3: Freezing parameters
The main part of the backbone model is not automatically frozen (We may add the option in future). To freeze the main part of the backbone model except the trainable parts (usually the delta paramters), use [freeze_module](opendelta.basemodel.DeltaBase.freeze_module) method. The `exclude` field obeys the same name-based addressing rules as the `modified_modules` field.

```python
# continue with the BART example
delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
delta_model.log()
```
````{collapse} <span style="color:rgb(141, 99, 224);font-weight:bold;font-style:italic">Click to view output</span>
```{figure} ../imgs/afterfreeze.png
---
width: 600px
name: afterfreeze
---
```
````
The `set_state_dict=True`  will tell the method to change the `state_dict` of the `backbone_model` to maintaining only the trainable parts. 


## STEP 4: Normal training pipeline

The **model** then can be trained in traditional training scripts. Two things should be noticed:

:::{admonition} Note
:class: note
1. No need to change the optimizer, since the optimizer will only calculated and store gradient for those parameters with `requires_grad=True`, and the `requires_grad` attribute has been changed during the call to [freeze_module](opendelta.basemodel.DeltaBase.freeze_module) method.
2. `model.eval()` or `model.train()` should be used when needed to set dropout, etc. Delta model doesn't touch those configuration.
:::
## STEP 5: Saved/Share the Delta Model

<img src="../imgs/hint-icon-2.jpg" height="30px"> *see [Save a delta model to local, or share with the community](saveload).*




