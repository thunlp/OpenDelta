
# from dataclasses import dataclass, field, fields
# from tkinter.messagebox import NO
# from typing import Optional, List, Union
# from xml.sax import default_parser_list
# from opendelta.utils.logging import get_logger

# logger = get_logger(__name__)


# alternative_names = {
#     "train_tasks":  ["train_tasks", "train_task", "task_name"],
# }


# @dataclass
# class DeltaCenterArguments:
#     """
#     The arguments that are used to distinguish between different delta models on the DeltaCenter
#     """
#     name: str = field(default="",
#                         metadata={"help": "The name of the delta model checkpoint"}
#     )
#     backbone_model: str = field(default="",
#                                 metadata={"help": "The backbone model of the delta model"}
#     )
#     model_name_or_path: str = field(
#         default = None,
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     delta_type: str = field(
#         default=None,
#         metadata={"help": "the type of type model, e.g., adapter, lora, etc."}
#     )
#     train_tasks: Optional[Union[List[str], str]]= field(
#         default=None,
#         metadata={"help": "the task(s） that the delta is trained on"}
#     )
#     checkpoint_size: Optional[float] = field(
#         default=None,
#         metadata={"help": "the size of the checkpoint, in MB"}
#     )
#     test_tasks: Optional[Union[List[str], str]] = field(
#         default=None,
#         metadata={"help": "the task(s) that the delta is tested on"}
#     )
#     test_performance: Optional[float] = field(
#         default=None,
#         metadata={"help": "the performance of the model on the test set"}
#     )
#     trainable_ratio: Optional[float] = field(
#         default=None,
#         metadata={"help": "the ratio of trainable parameters in the model"}
#     )
#     delta_ratio: Optional[float] = field(
#         default=None,
#         metadata={"help": "the ratio of delta parameters in the model"}
#     )





# def create_repo_name(prefix="", center_args=None, **kwargs
#                          ):
#     r"""Currently, it's only a simple concatenation of the arguments.
#     """
#     if center_args is None:
#         center_args = create_delta_center_args(**kwargs)
#     reponame = prefix+"_"
#     reponame += center_args.model_name_or_path.split()[-1]+"_" if center_args.model_name_or_path is not None else ""
#     reponame += center_args.delta_type+"_" if center_args.delta_type is not None else ""

#     # tasks

#     if isinstance(center_args.train_tasks, list):
#         train_tasks = "+".join(center_args.train_tasks)
#     elif center_args.train_tasks is not None:
#         train_tasks = center_args.train_tasks
#     else:
#         logger.warning("train_tasks are not find in all arguments. Do you miss them?")
#         train_tasks = None
#     reponame += train_tasks+"_" if train_tasks is not None else ""
#     reponame = reponame.strip("_")
#     return reponame

# def create_delta_center_args(**kwargs):
#     mdict = {}
#     field = fields(DeltaCenterArguments)
#     for f in field:
#         if f.name in kwargs:
#             mdict[f.name] = kwargs[f.name]
#         else:
#             for altername in alternative_names[f.name]:
#                 if altername in kwargs:
#                     mdict[f.name] = kwargs[altername]
#                     break
#     center_args = DeltaCenterArguments(**mdict)
#     return  center_args