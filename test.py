from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
from opendelta import Visualization
Visualization(model).structure_graph()
from opendelta import SplitModel
delta_model = SplitModel(model)
print("attach here")
delta_model.attach("output.dense", "adapter_A", "adapter")
delta_model.attach("output.dense", "adapter_B", "adapter")
delta_model.attach(["2.output.dense", "2.attention.output.dense"], "adapter_C", "adapter")
delta_model.update()
# delta_model.attach(["attention.self.query", "attention.self.key"], "lora_A", "lora")
delta_model.log() # This will visualize the backbone after modification and other information.
import torch
x = torch.randint(0, 10, (16, 128)).cuda()
import time
model = model.cuda()
st_time = time.time()
print("run here")
y = model(x)
print("detach here")
delta_model.detach("3.output.dense", "adapter_A")
delta_model.log()
print("run here")
y = model(x)
print(time.time() - st_time)