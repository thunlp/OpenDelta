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
import itertools


logger = logging.getLogger(__name__)



from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from typing import List, Dict
from collections import defaultdict
from openprompt.utils import round_list
import warnings
class MLMTokenizerWrapper:
    def __init__(self, max_seq_length, tokenizer, truncate_method):
        self.max_seq_length=max_seq_length
        self.tokenizer=tokenizer
        self.num_special_tokens_to_add = len(tokenizer("")['input_ids'])
        # from IPython import embed; embed(header="Truega")
        self.truncate_method=truncate_method
        self.total_passed_sentences = 0
        self.num_truncated_sentences = 0
        if truncate_method=='tail':
            self.truncate_fct = self.truncate_from_tail
        elif truncate_method=='head':
            self.truncate_fct = self.truncate_from_head
        elif truncate_method == 'balanced':
            self.truncate_fct = self.balanced_truncate
        else:
            raise NotImplementedError





    def merge_wrapped_example(self, wrapped_example, ):
        ''' # TODO doens't consider the situation that input has two parts
        '''

        wrapped_example, others = wrapped_example

        # for some dataset like SuperGLUE.COPA, the answer requires prediction an span of
        # the input. Or in generation tasks, we need to generate a piece of target_text.
        # In these case, it tokenized to the encoded_tgt_text for furture use.



        encoder_inputs = defaultdict(list)
        for piece in wrapped_example:
            encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False, return_special_tokens_mask=True )
            encoder_inputs['input_ids'].append(encode_text)
            encoder_inputs['shortenable_ids'].append([piece['shortenable_ids']] * len(encode_text))


        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        decoded_inputs = self.tokenizer.decode(encoder_inputs['input_ids'], clean_up_tokenization_spaces=False)

        # again_encode = self.tokenizer.encode(decoded_inputs, add_special_tokens=False, return_special_tokens_mask=True)
        # if len(again_encode)> self.max_seq_length - 2:
        #     print("length exceed!")
        #     print(wrapped_example)
        #     print(encoder_inputs['input_ids'])
        #     print(again_encode)
        #     print(decoded_inputs)
        #     exit()






        # delete shortenable ids

        # encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        # encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        # # create special input ids
        # encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        # # padding
        # encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        return decoded_inputs

    @staticmethod
    def balanced_truncate(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        '''truncate the inputs with balance, number of cut tokens is proportional to the part's length.
        '''
        shortenable_lens = [len(parts) if parts[0]==1 else 0
                                  for parts in input_dict['shortenable_ids']]
        total_shortenable_len = sum(shortenable_lens)
        num_tokens_to_truncate_each_part = [part_len/total_shortenable_len*num_tokens_to_truncate
                                                for part_len in shortenable_lens]
        round_list(num_tokens_to_truncate_each_part, num_tokens_to_truncate)

        truncated_example = defaultdict(list)
        for key in input_dict:
            parts = input_dict[key]
            for num_tokens_to_truncate_part, part in zip(num_tokens_to_truncate_each_part, parts):
                truncated_example[key].append(part[:len(part)-num_tokens_to_truncate_part])
        return truncated_example

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the rear
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']

        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts[::-1]):
                if len(part) == 0: # to prevent some part are empty after tokenization
                    continue
                if shortenable_ids[-1-i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[-1-i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def truncate_from_head(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the head
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts):
                if shortenable_ids[i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict

    # @staticmethod
    # def padding(input_dict: Dict,
    #             max_len: int, pad_id_for_inputs: int=0, pad_id_for_others: int=0) -> None:
    #     for key, value in input_dict.items():
    #         if (len(input_dict[key]) > max_len):
    #             raise ValueError(f'''
    #                 Truncated seq length of '{key}' still greater than max length '{max_len}.'
    #                 One possible reason is that no enough shortenable parts in template. Try add {{"shortenable": "True"}} property.
    #             ''')
    #         if 'input' in key:
    #             input_dict[key].extend([pad_id_for_inputs]*(max_len-len(value)))
    #         else:
    #             input_dict[key].extend([pad_id_for_others]*(max_len-len(value)))
    #     return input_dict


    # def add_special_tokens(self, encoder_inputs):
    #         # add special tokens
    #     for key in encoder_inputs:
    #         if key == "input_ids":
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore")
    #                 encoder_inputs[key] = self.tokenizer.build_inputs_with_special_tokens(
    #                                                     encoder_inputs[key])
    #     return encoder_inputs

    def truncate(self, encoder_inputs):
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        # print("num_specials", num_specials)
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences+=1
        if num_tokens_to_truncate>0:
            self.num_truncated_sentences += 1
            if num_tokens_to_truncate > sum([len(x) for x in encoder_inputs['shortenable_ids']]):
                raise RuntimeError("num_tokens_to_truncate larger than number of shortenable tokens.")
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                          num_tokens_to_truncate=num_tokens_to_truncate)
        return encoder_inputs













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

    def __init__(self, config, data_args, tokenizer, predict_with_generate, seed=42, default_max_length=1):
        self.config = config
        self.seed = seed
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.predict_with_generate = predict_with_generate
        self.default_max_length = default_max_length
        self.truncate_method = getattr(data_args, "truncate_method", "balanced")

        tid = getattr(config, "template_id", 0)
        vid = getattr(config, "verbalizer_id", 0)

        self.template = ManualTemplate(tokenizer=self.tokenizer, text = self.templates_text[tid])
        self.verbalizer = ManualVerbalizer(tokenizer=self.tokenizer, classes = self.labels_list, label_words=self.verbalizers[vid])

        # if self.predict_with_generate:
        #     self.reverse_verbalizer = {(int(x) for x in self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.verbalizer[label]))): label for label in self.labels_list}
        # else:
        #     self.reverse_verbalizer = {int(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.verbalizer[label]))[0]): label for label in self.labels_list}

        self.tokenizer_wrapper = MLMTokenizerWrapper(max_seq_length=self.data_args.max_source_length, tokenizer=self.tokenizer, truncate_method=self.truncate_method)

        generation_paradigm = getattr(config, "generation_paradigm", True)
        # self.prompt = PromptCollections[self.name](tid, vid, generation_paradigm)
        self.max_target_length = self.get_max_target_length(self.default_max_length)

    def get_max_target_length(self, default_max_length):
        if self.predict_with_generate:
            return max([len(label) for key, label in self.verbalizer.label_words_ids.items()])
        else:
            return default_max_length

    def seq2seq_format(self, source, target, extra_fields={}
                       ):

        return {'source': ' '.join(source),
                'target': ' '.join(target),
                'task': self.name,
                'extra_fields': extra_fields
                }

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


    def map_dataset(self, dataset):
        # from IPython import embed; embed(header="in get target length")
        return dataset.map(self.preprocessor).map(self.tokenizer_preprocessor)

    def preprocessor(self, example):
        return example

    def tokenizer_preprocessor(self, example):
        # source, target = example
        # from IPython import embed; embed(header="Trehre2")
        label = example['label']
        guid = example['idx']
        meta = dict(example)
        meta.pop("label")
        meta.pop("idx")



        # from IPython import embed; embed(header="Trehre2")

        e = InputExample(**{"meta": meta, 'label': label, 'guid': guid})
        template_e = self.template.wrap_one_example(e)
        encoded_sentence = self.tokenizer_wrapper.merge_wrapped_example(template_e)
        if self.predict_with_generate:
            # return {"source": encoded_sentence, 'target': ', 'extra_fields':[]}
            raise NotImplementedError
        else:
            return {"source": encoded_sentence, "label": label, 'target': '', 'extra_fields':{'dataset_name':self.name}}


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
        return self.map_dataset(dataset)

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


class MRPC(AbstractTask):
    name = "mrpc"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc', split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["sentence1:", example['sentence1'],
    #                  "sentence2:", example["sentence2"]]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLA(AbstractTask):
    name = "cola"
    labels_list = ["0", "1"]
    metric = [metrics.matthews_corrcoef]
    metric_names = ["matthews_correlation"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, script_version="master")

    # def preprocessor(self, example, add_prefix=True):
    #     src_texts = ["sentence:", example['sentence']]
    #     tgt_texts = [str(example['label'])]
    #     return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2(AbstractTask):
    name = "sst2"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = [
        
    ]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSB(AbstractTask):
    name = "stsb"
    labels_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    metric = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    metric_names = ["pearson", "spearmanr"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQP(AbstractTask):
    name = "qqp"
    labels_list = ["0", "1"]
    metric = [metrics.f1_score, metrics.accuracy]
    metric_names = ["f1", "accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLI(AbstractTask):
    name = "mnli"
    labels_list = ["0", "1", "2"]
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]


    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLI(AbstractTask):
    name = "qnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, script_version="master")

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

#Tested
class RTE(AbstractTask):
    name = "rte"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}


    templates_text = [
        """sentence1: {"meta": 'sentence1', "shortenable":True}. sentence2:,"""+
        """{"meta":"sentence2", "shortenable":True}. The answer was {"mask"}.""",
        ]

    verbalizers = [{
            "0": "yes",
            "1": "no"
        }]

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.rte")[split]
        else:
            return datasets.load_dataset('glue', 'rte',
                                     split=split, script_version="master")


#Tested
class SuperGLUEBoolQ(AbstractTask):
    name="superglue-boolq"
    labels_list = ['0', '1']
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = [
        {
            "0": "no",
            "1": "yes"
        },
    ]
    mlmhead_verbalizers = {
        "0": "no",
        "1": "yes"
    }
    templates_text = [
        """hypothesis: {"meta": "question", "shortenable":True} premise: {"meta":"passage", "shortenable":True} The answer was {"mask"}."""
    ]

    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/super_glue.boolq")[split]
        else:
            return datasets.load_dataset('super_glue', 'boolq', split=split, script_version="master")




class WNLI(AbstractTask):
    name = "wnli"
    labels_list = ["0", "1"]
    metric = [metrics.accuracy]
    metric_names = ["accuracy"]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    verbalizers = [{
        "0": "True",
        "1": "False",
    }]
    templates_text = [
        """{"meta": 'sentence1',"shortenable":True} Does it mean the following: "{"meta":'sentence2'}"? {"mask"}."""
    ]


    def load_dataset(self, split):
        if self.data_args.datasets_load_from_disk:
            return datasets.load_from_disk(f"{self.data_args.datasets_saved_path}/glue.wnli")[split]
        else:
            return datasets.load_dataset('glue', 'wnli', split=split, script_version="master")


#
class SuperGLUECB(AbstractTask):
    name = "superglue-cb"
    labels_list = ['0', '1', '2']
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metric = [metrics.mean_multiclass_f1(num_classes=3), metrics.accuracy]
    metric_names = ["f1_multiclass", "accuracy"]

    verbalizers = [{
        "0": "yes",
        "1": "no",
        "2": "maybe"
    }]
    templates_text = [
        """hypothesis: {"meta": 'hypothesis',"shortenable":True} premise: {"meta":'premise', "shortenable":True} The answer was {"mask"}."""
    ]

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

    verbalizers = [{
        "0": "1",
        "1": "2",
    }]
    templates_text = [
        """choice1: {"meta":"choice1"} choice2: {"meta":"choice2"} premise: {"meta":"premise", "shortenable":True} The {"meta":"question"} answer was choice{"mask"}."""
    ]

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

    # generation_verbalizers = [{
    #     "0": "no",
    #     "1": "yes",
    # },
    # ]

    verbalizers = [{
        "0": "no",
        "1": "yes",
    }]
    templates_text = [
        """question: {"meta":"question", "shortenable":False} answer: {"meta":"answer", "shortenable":False, "post_processing": lambda x:x+"."} paragraph: {"meta":"paragraph", "shortenable":True} The answer was {"mask"}."""
    ]


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

    verbalizers = [{
        "0": "No",
        "1": "Yes",
    }]

    templates_text = [
        """sentence1: {"meta":"sentence1"} sentence2: {"meta":"sentence2", "shortenable": True} word: {"meta":"word"} {"mask"}.
        """
    ]

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
#         return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


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
    def get(self, task, config, data_args, tokenizer,predict_with_generate, seed=42):
        if task in TASK_MAPPING:
            return TASK_MAPPING[task](config, data_args, tokenizer,predict_with_generate, seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
