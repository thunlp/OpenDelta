from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
from opendelta import Visualization
Visualization(model).structure_graph()
from opendelta import SplitModel
delta_model = SplitModel(model)

print("split_attach here")
delta_model.split_attach("output.dense", "adapter_A", "adapter", bottleneck_dim=12)
delta_model.split_attach("output.dense", "adapter_B", "adapter", bottleneck_dim=16, non_linearity="relu")
# delta_model.split_attach(["2.output.dense", "2.attention.output.dense"], "adapter_C", "adapter")
# delta_model.split_attach(["attention.self.query", "attention.self.key"], "lora_A", "lora", r=8)
delta_model.update()
delta_model.log() # This will visualize the backbone after modification and other information.

print("batchsplit_attach here")
delta_model.split_attach(["attention.self.query", "attention.self.key"], "lora_A", "batch_lora", r=4)
delta_model.split_attach(["attention.self.query", "attention.self.key"], "lora_B", "batch_lora", r=8)
# delta_model.split_attach(["attention.self.query", "attention.self.key"], "adapter_E", "batch_adapter")
# delta_model.split_attach(["attention.self.query", "attention.self.key"], "adapter_F", "batch_adapter")
delta_model.update()
delta_model.log()

print("split_detach and save here")
delta_model.save_split("adapter_A", "adapter_A_split.pt")
delta_model.split_detach("adapter_A")
delta_model.save_split("lora_A", "lora_A_split.pt")
delta_model.split_detach("lora_A")
delta_model.update()
delta_model.log() # This will visualize the backbone after modification and other information.

print("load back here")
delta_model.load_split("adapter_A", "adapter_A_split.pt")
delta_model.load_split("lora_A", "lora_A_split.pt")
delta_model.update()
delta_model.log() # This will visualize the backbone after modification and other information.

print("run here")
import torch
x = torch.randint(0, 10, (16, 128)).cuda()
delta_model.set_batchsplit_pattern(['lora_A']*4 + ['lora_B']*12)
model = model.cuda()
y = model(x)
