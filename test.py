from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
from opendelta import Visualization
Visualization(model).structure_graph()
from opendelta import SplitModel
delta_model = SplitModel(model, batch_size=[1]*16, modified_modules=['output.dense'])
delta_model.log() # This will visualize the backbone after modification and other information.
import torch
x = torch.randint(0, 10, (16, 128)).cuda()
import time
model = model.cuda()
st_time = time.time()
for t in range(10):
    y = model(x)
print(time.time() - st_time)