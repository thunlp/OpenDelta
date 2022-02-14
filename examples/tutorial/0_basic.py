from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")
# suppose we load BART

from opendelta import Visualization
print("before modify")
Visualization(model).structure_graph()
"""
The white part is the name of the module.
The green part is the module's type.
The blue part is the tunable parameters, i.e., the parameters that require grad computation.
The grey part is the frozen parameters, i.e., the parameters that do not require grad computation.
The red part is the structure that is repeated and thus folded.
The purple part is the delta parameters inserted into the backbone model.
"""

from opendelta import LoraModel
delta_model = LoraModel(backbone_model=model, modified_modules=['fc2'])
print("after modify")
delta_model.log()
# This will visualize the backbone after modification and other information.

delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
print("after freeze")
delta_model.log()
# The set_state_dict=True will tell the method to change the state_dict of the backbone_model to maintaining only the trainable parts.
