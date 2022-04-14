import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq


@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   def check_uniqueness(self, samples):
        assert len(np.unique(samples)) == 1

   def __call__(self, features):
     #    tasks = [d.pop('task') for d in features]
     #    self.check_uniqueness(tasks)
        output = super().__call__(features)
     #    output["task"] = tasks[0]
        return output

# class CustomDataCollator(DefaultDataCollator):
#    def check_uniqueness(self, samples):
#         assert len(np.unique(samples)) == 1

#    def __call__(self, features):
#         mask_positions = [d.pop('mask_positions') for d in features]
#      #    self.check_uniqueness(tasks)
#         output = super().__call__(features)

#      #    output["task"] = tasks[0]
#         return output