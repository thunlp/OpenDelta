import bmtrain as bmt
import opendelta as od
from opendelta import LoraModel, AdapterModel, CompacterModel, LowRankAdapterModel, BitFitModel
import torch
import numpy
import random

def manual_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

from model_center.model import Bert, BertConfig
bmt.init_distributed()
config = BertConfig.from_pretrained("/yinxr/zwl/.cache/model_center/bert-base-uncased")
config.dropout_p = 0
model = Bert.from_pretrained("/yinxr/zwl/.cache/model_center/bert-base-uncased", config)

print("before modify")
od.Visualization(model).structure_graph()

manual_seed(233)
delta_model = LoraModel(backbone_model=model, modified_modules=['project_q', 'project_k'])
# delta_model = AdapterModel(backbone_model=model, modified_modules=['[r]layers\\.(\d)+\\.self_att', '[r]layers\\.(\d)+\\.ffn'])
# delta_model = CompacterModel(backbone_model=model, modified_modules=['[r]layers\\.(\d)+\\.self_att', '[r]layers\\.(\d)+\\.ffn'])
# delta_model = LowRankAdapterModel(backbone_model=model, modified_modules=['[r]layers\\.(\d)+\\.self_att', '[r]layers\\.(\d)+\\.ffn'])
# delta_model = BitFitModel(backbone_model=model, modified_modules=['[r]layers\\.(\d)+\\.self_att', '[r]layers\\.(\d)+\\.ffn', '[r](.*)layernorm(.*)'])

print(delta_model.delta_modules)

print("after modify")
delta_model.log()
# This will visualize the backbone after modification and other information.

delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
print("after freeze")
delta_model.log()
# The set_state_dict=True will tell the method to change the state_dict of the backbone_model to maintaining only the trainable parts.

manual_seed(233)
inp = torch.randint(0, 30000, (32, 128)).cuda()
length = torch.randint(0, 128, (32,)).cuda()
attention_mask = (torch.arange(inp.shape[1], device=inp.device)[None, :].repeat(inp.shape[0], 1) < length[:, None])
out = model(inp, attention_mask=attention_mask, output_logits=True).logits
print(out)

if bmt.rank() == 0:
    torch.save(model.state_dict(), "test.pt")
    ckpt = torch.load("test.pt")
    print(ckpt.keys())