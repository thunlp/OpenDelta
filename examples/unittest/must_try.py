# use tranformers as usual.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
# A running example
inputs_ids = t5_tokenizer.encode("Is Harry Poter wrtten by JKrowling", return_tensors="pt")
t5_tokenizer.decode(t5.generate(inputs_ids)[0]) 
# >>> '<pad><extra_id_0>? Is it Harry Potter?</s>'


# use existing delta models
from opendelta import AutoDeltaModel, AutoDeltaConfig
# use existing delta models from DeltaCenter
delta = AutoDeltaModel.from_finetuned("thunlp/Spelling_Correction_T5_LRAdapter_demo", backbone_model=t5)
# freeze the whole backbone model except the delta models.
delta.freeze_module()
# visualize the change
delta.log()


t5_tokenizer.decode(t5.generate(inputs_ids)[0]) 
# >>> <pad> Is Harry Potter written by JK Rowling?</s>


# Now save merely the delta models, not the whole backbone model, to tmp/
delta.save_finetuned(".tmp")
import os; os.listdir(".tmp")
# >>>  The state dict size is 1.443 MB
# >>>  We encourage users to push their final and public models to delta center to share them with the community!


# reload the model from local url and add it to pre-trained T5.
t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
delta1 = AutoDeltaModel.from_finetuned(".tmp", backbone_model=t5)
import shutil; shutil.rmtree(".tmp") # don't forget to remove the tmp files. 
t5_tokenizer.decode(t5.generate(inputs_ids)[0]) 
# >>> <pad> Is Harry Potter written by JK Rowling?</s>

# detach the delta models, the model returns to the unmodified status.
delta1.detach()
t5_tokenizer.decode(t5.generate(inputs_ids)[0])  
# >>> '<pad><extra_id_0>? Is it Harry Potter?</s>'

# use default configuration for cunstomized wrapped models which have PLMs inside. This is a common need for users. 
import torch.nn as nn
class WrappedModel(nn.Module):
  def __init__(self, inner_model):
    super().__init__()
    self.inner = inner_model
  def forward(self, *args, **kwargs):
    return self.inner(*args, **kwargs)

wrapped_model = WrappedModel(WrappedModel(t5))

# say we use LoRA
delta_config = AutoDeltaConfig.from_dict({"delta_type":"lora"})
delta2 = AutoDeltaModel.from_config(delta_config, backbone_model=wrapped_model)
delta2.log()
# >>> root
#       -- inner
#          -- inner
#             ...
#             ... lora_A:[8,1024], lora_B:[1024,8]
delta2.detach()

# use a not default configuration
# say we add lora to the last four layer of the decoder of t5, with lora rank=5
delta_config3 = AutoDeltaConfig.from_dict({"delta_type":"lora", "modified_modules":["[r]decoder.*((20)|(21)|(22)|(23)).*DenseReluDense\.wi"], "lora_r":5})
delta3 = AutoDeltaModel.from_config(delta_config3, backbone_model=wrapped_model)
delta3.log()


