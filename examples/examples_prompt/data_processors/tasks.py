from collections import OrderedDict
import collections
import abc
import functools
from selectors import EpollSelector
from typing import Callable, List, Mapping
from examples_prompt.trainers.trainer_utils import pad_punctuation
from examples_prompt.metrics import metrics
from .utils import round_stsb_target
import datasets
import logging
import numpy as np
import torch
import re
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt.plms.utils import TokenizerWrapper
from openprompt.data_utils import InputExample
from openprompt.prompts import GenerationVerbalizer
import itertools


logger = logging.getLogger(__name__)



from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from typing import List, Dict
from collections import defaultdict
from openprompt.utils import round_list
import warnings

# class MLMTokenizerWrapper:
#     def __init__(self, max_seq_length, tokenizer, truncate_method, mask_token_func=lambda i: "<mask>"):
#         self.max_seq_length=max_seq_length
#         self.tokenizer=tokenizer
#         self.num_special_tokens_to_add = len(tokenizer("")['input_ids'])
#         # from IPython import embed; embed(header="Truega")
#         self.truncate_method=truncate_method
#         self.total_passed_sentences = 0
#         self.num_truncated_sentences = 0
#         self.mask_token_func = mask_token_func

#         if truncate_method=='tail':
#             self.truncate_fct = self.truncate_from_tail
#         elif truncate_method=='head':
#             self.truncate_fct = self.truncate_from_head
#         elif truncate_method == 'balanced':
#             self.truncate_fct = self.balanced_truncate
#         else:
#             raise NotImplementedError


#     def merge_wrapped_example(self, wrapped_example,):
#         ''' # TODO doens't consider the situation that input has two parts
#         '''

#         wrapped_example

#         # for some dataset like SuperGLUE.COPA, the answer requires prediction an span of
#         # the input. Or in generation tasks, we need to generate a piece of target_text.
#         # In these case, it tokenized to the encoded_tgt_text for furture use.



#         encoder_inputs = defaultdict(list)
#         # from IPython import embed; embed(header="Line 67")

#         mask_count = 0
#         for piece in wrapped_example:
#             if piece['text'] == "<mask>":
#                 encode_text = self.tokenizer.encode(self.mask_token_func(mask_count), add_special_tokens=False, return_special_tokens_mask=True )
#                 mask_count += 1
#             else:
#                 encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False, return_special_tokens_mask=True )
#             encoder_inputs['input_ids'].append(encode_text)
#             encoder_inputs['shortenable_ids'].append([piece['shortenable_ids']] * len(encode_text))


#         encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
#         encoder_inputs.pop("shortenable_ids")
#         encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
#         decoded_inputs = self.tokenizer.decode(encoder_inputs['input_ids'], clean_up_tokenization_spaces=False)

#         return decoded_inputs


#     @staticmethod
#     def balanced_truncate(input_dict: Dict,
#                  num_tokens_to_truncate: int=0) -> Dict:
#         '''truncate the inputs with balance, number of cut tokens is proportional to the part's length.
#         '''
#         shortenable_lens = [len(parts) if parts[0]==1 else 0
#                                   for parts in input_dict['shortenable_ids']]
#         total_shortenable_len = sum(shortenable_lens)
#         num_tokens_to_truncate_each_part = [part_len/total_shortenable_len*num_tokens_to_truncate
#                                                 for part_len in shortenable_lens]
#         round_list(num_tokens_to_truncate_each_part, num_tokens_to_truncate)

#         truncated_example = defaultdict(list)
#         for key in input_dict:
#             parts = input_dict[key]
#             for num_tokens_to_truncate_part, part in zip(num_tokens_to_truncate_each_part, parts):
#                 truncated_example[key].append(part[:len(part)-num_tokens_to_truncate_part])
#         return truncated_example

#     @staticmethod
#     def truncate_from_tail(input_dict: Dict,
#                  num_tokens_to_truncate: int=0) -> Dict:
#         r"""truncate the inputs from the rear
#         """
#         truncated_example = defaultdict(list)
#         shortenable_ids = input_dict['shortenable_ids']

#         for key in input_dict:
#             parts = input_dict[key]
#             to_trunc = num_tokens_to_truncate
#             for i, part in enumerate(parts[::-1]):
#                 if len(part) == 0: # to prevent some part are empty after tokenization
#                     continue
#                 if shortenable_ids[-1-i][0]==0: # ==0 means the part is not shortenable
#                     continue
#                 parts[-1-i] = part[:-to_trunc] if to_trunc<len(part) else []
#                 to_trunc -= len(part)
#                 if to_trunc <= 0:
#                     break
#             truncated_example[key] = parts
#         return truncated_example

#     @staticmethod
#     def truncate_from_head(input_dict: Dict,
#                  num_tokens_to_truncate: int=0) -> Dict:
#         r"""truncate the inputs from the head
#         """
#         truncated_example = defaultdict(list)
#         shortenable_ids = input_dict['shortenable_ids']
#         for key in input_dict:
#             parts = input_dict[key]
#             to_trunc = num_tokens_to_truncate
#             for i, part in enumerate(parts):
#                 if shortenable_ids[i][0]==0: # ==0 means the part is not shortenable
#                     continue
#                 parts[i] = part[:-to_trunc] if to_trunc<len(part) else []
#                 to_trunc -= len(part)
#                 if to_trunc <= 0:
#                     break
#             truncated_example[key] = parts
#         return truncated_example

#     @staticmethod
#     def concate_parts(input_dict: Dict) -> Dict:
#         for key in input_dict:
#             input_dict[key] = list(itertools.chain(*input_dict[key]))
#         return input_dict


#     def truncate(self, encoder_inputs):
#         total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
#         num_specials = self.num_special_tokens_to_add
#         # print("num_specials", num_specials)
#         num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
#         self.total_passed_sentences+=1
#         if num_tokens_to_truncate>0:
#             self.num_truncated_sentences += 1
#             if num_tokens_to_truncate > sum([len(x) for x in encoder_inputs['shortenable_ids']]):
#                 raise RuntimeError("num_tokens_to_truncate larger than number of shortenable tokens.")
#             encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
#                           num_tokens_to_truncate=num_tokens_to_truncate)
#         return encoder_inputs

#     def tokenizer_preprocessor(self, example):
#         # source, target = example
#         # from IPython import embed; embed(header="Trehre2")
#         label = example['label']
#         guid = example['idx']
#         meta = dict(example)
#         meta.pop("label")
#         meta.pop("idx")



#         # from IPython import embed; embed(header="Trehre2")

#         e = InputExample(**{"meta": meta, 'label': label, 'guid': guid})

#         if self.predict_with_generate:
#             e = self.verbalizer.wrap_one_example(e)
#         example_wrapped = self.template.wrap_one_example(e)
#         encoded_sentence = self.tokenizer_wrapper.merge_wrapped_example(example_wrapped)
#         print(encoded_sentence)
#         if self.predict_with_generate:
#             # return {"source": encoded_sentence, 'target': ', 'extra_fields':[]}
#             return {"source": encoded_sentence, "label": label, 'target': '', 'extra_fields':{'dataset_name':self.name}}
#         else:
#             return {"source": encoded_sentence, "label": label, 'target': e.target_text, 'extra_fields':{'dataset_name':self.name}}













class AbstractTask(abc.ABC):
    name = NotImplemented
    config = NotImplemented
    prefix = NotImplemented
    metric = NotImplemented
    metric_names = NotImplemented
    split_map = None
    labels_list = None
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}
    small_datasets_without_all_splits = ["cola", "wnli", "rte", "superglue-cb", "superglue-copa", "superglue-multirc",
                                         "superglue-wic", "superglue-wsc.fixed", "superglue-rte", "mrpc", "stsb",
                                         "superglue-boolq", "qqp", "qnli", "superglue-record", "sst2"]
    large_data_without_all_splits = [] #["qqp", "qnli", "superglue-record", "sst2"]

    def __init__(self, config, data_args, seed=42, default_max_length=1):
        self.config = config
        self.seed = seed
        self.data_args = data_args
        # self.tokenizer = tokenizer
        # self.predict_with_generate = predict_with_generate
        self.default_max_length = default_max_length

        # generation_paradigm = getattr(config, "generation_paradigm", True)
        # self.prompt = PromptCollections[self.name](tid, vid, generation_paradigm)


    # def get_max_target_length(self, default_max_length):
    #     if self.predict_with_generate:
    #         return -1
    #     else:
    #         return default_max_length

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def shuffled_indices(self, dataset):
        num_samples = len(dataset)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return torch.randperm(num_samples, generator=generator).tolist()

    def subsample(self, dataset, n_obs=None, indices=None):
        """
        Given a dataset returns the subsampled dataset.
        :param n_obs: the number of samples of the subsampled dataset.
        :param indices: indices to select the samples from, if not given, indices are computed
        from by shuffling the given dataset.
        :return: subsampled dataset.
        """
        num_samples = len(dataset)
        n_obs = self.check_n_obs(n_obs, num_samples)
        if indices is None:
           indices = self.shuffled_indices(dataset)
        indices = indices[:n_obs]
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, self.config, split=split, script_version="master")

    def get_split_indices(self, split, dataset, validation_size):
        indices = self.shuffled_indices(dataset)
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def preprocessor(self, example):
        return example

    def get(self, split, n_obs=None, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split in ["eval", "dev", "valid"]:
            split = "validation"
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_split_indices(split, dataset, validation_size=len(dataset)//2)
            dataset = self.subsample(dataset, n_obs, indices)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_split_indices(split, dataset, validation_size=1000)
            dataset = self.subsample(dataset, n_obs, indices)
        else:
            mapped_split = self.split_to_data_split[split]
            dataset = self.load_dataset(split=mapped_split)
            # shuffles the data and samples it.
            if n_obs is not None:
                dataset = self.subsample(dataset, n_obs)
        return dataset.map(self.preprocessor)

class Squad(AbstractTask):
    name = "squad"
    metric = [metrics.squad]

    def load_dataset(self, split):
        return datasets.load_dataset(self.name, split=split, script_version="master")

    def preprocessor(self, example, add_prefix):
        answer = pad_punctuation(example['answers']['text'][0])
        question = pad_punctuation(example['question'])
        context = pad_punctuation(example['context'])
        source = ["question:", question,
                  "context:", context]
        target = [answer]
        return self.seq2seq_format(source, target, add_prefix)


##GLUE
class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {"0": """sentence: {"meta": 'sentence', "shortenable":True} Are there any error in the sentence? {"mask"}""",
    }

    verbalizers = {
        "0":{ "0": "yes", "1": "no"}
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.cola")[split]
        else:
            return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0":{"0":"negative","1":"positive"}
    }

    templates_text = {
        "0":"""The sentiment of sentence: "{"meta":"sentence", "shortenable":True} is {"mask"}."""
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.sst2")[split]
        else:
            return datasets.load_dataset('glue', 'sst2',
                                        split=split, script_version="master")



class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    templates_text = {
        "0": """sentence1: {"meta": 'sentence1', "shortenable":True}. sentence2: {"meta":"sentence2", "shortenable":True}. Are sentence1 and sentence2 equivalent? {"mask"}.""",
    }

    verbalizers = {
        "0":{"0": "no","1": "yes"}
    }




    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.mrpc")[split]
        else:
            return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")



class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {"0":
        """question1: {"meta": 'question1', "shortenable":True}. question2: {"meta": 'question2', "shortenable":True} Are question1 and question2 equivalent? {"mask"}."""
    }

    verbalizers = {
        "0":{"0": "no","1": "yes"}
    }


    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.qqp")[split]
        else:
            return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")



class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    verbalizers = {
        ""
    }

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]


    templates_text = {
        "0":"""premise: {"meta": 'premise', "shortenable":True}. hypothesis: {"meta": 'hypothesis', "shortenable":True} Does the premise entails the hypothesis? {"mask"}.""",
    }

    verbalizers = {
        "0":{
            "0": "yes",
            "1": "neutral",
            "2": "no",
        }
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.mnli")[split]
        else:
            return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["premise:", example['premise'],
    #                  "hypothesis", example["hypothesis"]]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    templates_text = {
        "0": """premise: {"meta": 'sentence', "shortenable":True}. hypothesis: {"meta": 'question', "shortenable":True}"""+
        """Does the premise entails the hypothesis? {"mask"}.""",
    }

    verbalizers = {
        "0":{
            "0": "yes",
            "1": "no",
        }
    }


    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.qnli")[split]
        else:
            return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    # def load_dataset(self, split):
    #     return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["question:", example['question'],
    #                  "sentence:", example["sentence"]]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

#Tested
class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    templates_text = {
        "0": """sentence1: {"meta": 'sentence1', "shortenable":True} sentence2: {"meta":"sentence2", "shortenable":True} The answer was {"mask"}.""",
    }

    verbalizers = {
        "0":{"0": "yes",
            "1": "no"
        }
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.rte")[split]
        else:
            return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")



class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0":{"0": "True",
            "1": "False",
            }
    }
    templates_text = {"0": """{"meta": 'sentence1',"shortenable":True} Does it mean the following: "{"meta":'sentence2'}"? {"mask"}."""
    }


    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.wnli")[split]
        else:
            return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")


#SuperGLUE
class SuperGLUEBoolQ(AbstractTask):
    name="superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = {
        "0": {
            "0": "no",
            "1": "yes"
        },
    }

    templates_text = {
        "0": """hypothesis: {"meta": "question", "shortenable":True} premise: {"meta":"passage", "shortenable":True} The answer was {"mask"}."""
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.boolq")[split]
        else:
            return datasets.load_dataset('super_glue', 'boolq', split=split, script_version="master")


#
class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    verbalizers = {
        "0":{"0": "yes",
            "1": "no",
            "2": "maybe"
        }
    }
    templates_text = {
        "0": """hypothesis: {"meta": 'hypothesis',"shortenable":True} premise: {"meta":'premise', "shortenable":True} The answer was {"mask"}."""
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.cb")[split]
        else:
            return datasets.load_dataset('super_glue', 'cb', split=split, script_version="master")


class SuperGLUECOPA(AbstractTask):
    name = "superglue-copa"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "0":{
            "0": "1",
            "1": "2",
        }
    }
    templates_text = {
       "0": """choice1: {"meta":"choice1"} choice2: {"meta":"choice2"} premise: {"meta":"premise", "shortenable":True} The {"meta":"question"} answer was choice{"mask"}."""
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.copa")[split]
        else:
            return datasets.load_dataset('super_glue', 'copa', split=split, script_version="master")


class SuperGLUEMultiRC(AbstractTask):
    name = "superglue-multirc"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.f1_score,
              metrics.accuracy]
    metric_names = ["f1", "em"]


    verbalizers = {
        "0": {
        "0": "no",
        "1": "yes",
        }
    }
    templates_text = {
        "0": """question: {"meta":"question", "shortenable":False} answer: {"meta":"answer", "shortenable":False, "post_processing": lambda x:x+"."} paragraph: {"meta":"paragraph", "shortenable":True} The answer was {"mask"}."""
    }


    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.multirc")[split]
        else:
            return datasets.load_dataset('super_glue', 'multirc', split=split, script_version="master")

    def remove_markup(self, text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text

    def preprocessor(self, example):
        # T5 applies remove_markup to the joined string, but this should not make
        # any difference as well.
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/a1352e625db7ec114062f99d99b0565b9e45c155/t5/data/preprocessors.py#L797
        example["question"] = self.remove_markup(example["question"])
        example["answer"] = self.remove_markup(example["answer"])
        example["paragraph"] = self.remove_markup(example["paragraph"])
        return example



class SuperGLUEWIC(AbstractTask):
    name = "superglue-wic"
    labels_list = ['0', '1']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]

    verbalizers = {
        "0": {
        "0": "No",
        "1": "Yes",
        }
    }

    templates_text = {
        "0": """sentence1: {"meta":"sentence1"} sentence2: {"meta":"sentence2", "shortenable": True} word: {"meta":"word"} {"mask"}."""
    }

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.wic")[split]
        else:
            return datasets.load_dataset('super_glue', 'wic', split=split, script_version="master")



# class SuperGLUEWSCFixed(AbstractTask):
#     # source: https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py
#     """Convert WSC examples to text2text format.
#      WSC includes a sentence along with 2 'spans': the first denoting a noun and
#      the other a pronoun. The 'label' specifies whether or not the pronoun is
#      referencing the noun. This preprocessor puts ' * ' around the noun and ' # '
#      around the pronoun.
#      For example, a typical example from WSC might look like
#      {
#          'text': 'This is a test sentence .',
#          'span1_text': 'test',
#          'span1_index': 3,
#          'span2_text': 'This',
#          'span2_index': 0,
#          'label': 0
#      }
#      This example would be transformed to
#      {
#          'inputs': 'wsc text: # This # is a * test * sentence .',
#          'targets': 'False'
#      }
#     """
#     name = "superglue-wsc.fixed"
#     labels_list = ['0', '1']
#     split_to_data_split = {"train": "train",
#                            "validation": "validation",
#                            "test": "validation"}
#     metric = [metrics.accuracy]
#     metric_names = ["accuracy"]

#     def load_dataset(self, split):
#         return datasets.load_dataset('super_glue', 'wsc.fixed', split=split, script_version="master")

#     def _mark_span(self, text, span_str, span_idx, mark):
#         pattern_tmpl = r'^((?:\S+\s){N})(W)'
#         pattern = re.sub('N', str(span_idx), pattern_tmpl)
#         pattern = re.sub('W', span_str, pattern)
#         return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

#     def preprocessor(self, example, add_prefix=True):
#         # converts text as done in T5.
#         text = example['text']
#         text = self._mark_span(text, example['span1_text'], example['span1_index'], '*')
#         # Compensate for 2 added "words" added in previous step.
#         span2_index = example['span2_index'] + 2 * int(example['span1_index'] < example['span2_index'])
#         text = self._mark_span(text, example['span2_text'], span2_index, '#')
#         src_texts = ["text:", text]
#         tgt_texts = [str(example["label"])]
#         return self.fseq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERecord(AbstractTask):
    """Convert ReCoRD examples to text2text examples.
    ReCoRD contains a passage, query containing a '@placeholder' string, and a set
    of entities that are the possible values of the placeholder. Each train and
    validation example will have a list of answers, any of which would be
    considered correct.
    For example, a typical example from ReCoRD might look like
    {
      'passsage': 'This is the passage.',
      'query': 'A @placeholder is a bird.',
      'entities': ['penguin', 'potato', 'pigeon'],
      'answers': ['penguin', 'pigeon'],
    }
    which this preprocessor would turn into the following two examples:
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'penguin',
    }
    and
    {
      'inputs': 'record query: A @placeholder is a bird. entities: penguin, '
                'potato, pigeon passage: This is the passage.',
      'targets': 'pigeon',
    }
    """
    name = "superglue-record"
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.squad]
    metric_names = ["squad"]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'record', split=split, script_version="master")

    def preprocessor(self, batch, add_prefix=True):
        new_batch = collections.defaultdict(list)
        keys = batch.keys()
        for values in zip(*batch.values()):
            ex = {k: v for k, v in zip(keys, values)}
            # updates the passage.
            passage = ex['passage']
            passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
            passage = re.sub(r'\n@highlight\n', '. ', passage)
            inputs = f"record query: {ex['query']} entities: {', '.join(ex['entities'])} passage: {passage}"
            if add_prefix:
                inputs = self.name + " " + inputs
            # duplicates the samples based on  number of answers.
            num_answers = len(ex["answers"])
            num_duplicates = np.maximum(1, num_answers)
            new_batch["source"].extend([inputs] * num_duplicates)
            new_batch["target"].extend(ex["answers"] if num_answers > 0 else ["<unk>"])
            new_batch["task"].extend([self.name] * num_duplicates)
            new_batch["extra_fields"].extend([{"answers": ex["answers"]}]*num_duplicates)
        return new_batch

    def map_dataset(self, dataset, add_prefix=True):
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
            batched=True, remove_columns=dataset.column_names)


TASK_MAPPING = OrderedDict(
    [
        ('squad', Squad),
        ('mrpc', MRPC),
        ('cola', COLA),
        ('sst2', SST2),
        ('qnli', QNLI),
        ('rte', RTE),
        ('wnli', WNLI),
        ('mnli', MNLI),
        ('qqp', QQP),
        ('stsb', STSB),
        ('superglue-boolq', SuperGLUEBoolQ),
        ('superglue-cb', SuperGLUECB),
        ('superglue-copa', SuperGLUECOPA),
        ('superglue-multirc', SuperGLUEMultiRC),
        ('superglue-wic', SuperGLUEWIC),
        # ('superglue-wsc.fixed', SuperGLUEWSCFixed),
        ('superglue-record', SuperGLUERecord)
    ]
)

class AutoTask:
    @classmethod
    def get(self, task, config, data_args, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
