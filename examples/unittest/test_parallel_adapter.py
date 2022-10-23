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
delta_config = AutoDeltaConfig.from_dict({"delta_type":"parallel_adapter"})
delta2 = AutoDeltaModel.from_config(delta_config, backbone_model=wrapped_model)
delta2.log()
# >>> root
#       -- inner
#          -- inner
#             -- encoder
#                -- block
#                   -- 0
#                      -- layer
#                         ...
#                         -- parallel_adapter
#                         ...
#                  -- 1
#                     -- DenseRuleDense
#                        -- wi
#                           -- parallel_adapter
#                  ...
delta2.detach()
delta2.log()
