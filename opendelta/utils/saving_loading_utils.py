
from io import RawIOBase
import re
from tarfile import HeaderError
from typing import Dict, List, Union, Optional, Callable
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.model_md5 import gen_model_hash, gen_parameter_hash
import torch
import os
from opendelta import logging
import torch.nn as nn
from transformers.file_utils import (
    WEIGHTS_NAME,
    PushToHubMixin,
    is_offline_mode,
    is_remote_url,
    hf_bucket_url,
    cached_path,
    )
from transformers.utils.dummy_pt_objects import PreTrainedModel
import hashlib
try:
    from DeltaCenter import OssClient
except:
    pass
import  yaml
from dataclasses import dataclass, field, fields
import datetime

logger = logging.get_logger(__name__)


alternative_names = {
    "train_tasks":  ["train_tasks", "train_task", "task_name"],
}


@dataclass
class DeltaCenterArguments:
    """
    The arguments that are used to distinguish between different delta models on the DeltaCenter
    """
    name: str = field(default="",
                        metadata={"help": "The name of the delta model checkpoint"}
    )
    backbone_model: str = field(default="",
                                metadata={"help": "The backbone model of the delta model"}
    )
    model_path_public: str = field(
        default = None,
        metadata={"help": "Publicly available path (url) to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    delta_type: str = field(
        default=None,
        metadata={"help": "the type of type model, e.g., adapter, lora, etc."}
    )
    train_tasks: Optional[Union[List[str], str]]= field(
        default=None,
        metadata={"help": "the task(s） that the delta is trained on"}
    )
    checkpoint_size: Optional[float] = field(
        default=None,
        metadata={"help": "the size of the checkpoint, in MB"}
    )
    test_tasks: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "the task(s) that the delta is tested on"}
    )
    test_performance: Optional[float] = field(
        default=None,
        metadata={"help": "the performance of the model on the test set"}
    )
    test_metrics: Optional[str] = field(
        default=None,
        metadata={"help": "the metrics used by the model"}
    )
    trainable_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "the ratio of trainable parameters in the model"}
    )
    delta_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "the ratio of delta parameters in the model"}
    )



class SaveLoadMixin(PushToHubMixin):
    def add_configs_when_saving(self,):
        self.config.backbone_class = self.backbone_model.__class__.__name__
        self.config.backbone_checkpoint_name = os.path.split(self.backbone_model.config._name_or_path.strip("/"))[-1]
        self.config.backbone_hash = gen_model_hash(self.backbone_model)




    def save_finetuned(
        self,
        finetuned_delta_path: Optional[Union[str, os.PathLike]] = "./delta_checkpoints/",
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_dc: bool = True,
        center_args: Optional[Union[DeltaCenterArguments, dict]] = None,
        center_args_pool: Optional[dict] = None,
        list_tags: Optional[List] = None,
        dict_tags: Optional[Dict] = None,
        delay_push: bool = False,
        test_result = None
    ):
        r"""
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        :py:meth:`~DeltaBase.from_finetuned` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (:obj:`bool`, *optional*, defaults to :obj:`True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set ``save_config=True`` only on the main process
                to avoid race conditions.
            state_dict (nested dictionary of :obj:`torch.Tensor`):
                The state dictionary of the model to save. Will default to ``self.state_dict()``, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (:obj:`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace ``torch.save`` by another method.
            push_to_dc (:obj:`bool`, *optional*, defaults to :obj:`True`): Whether or not to push the model to the DeltaCenter.
            center_args (:obj:`Union[DeltaCenterArguments, dict]`, *optional*, defaults to :obj:`None`): The arguments
                that are used to distinguish between different delta models on the DeltaCenter. It has higher priority than the `center_args_pool`.
                It will be used to group delta models.
            center_args_pool (:obj:`dict`, *optional*, defaults to :obj:`None`): The arguments's pool for DeltaCenter
                Together with center_args, they are are used to distinguish between different delta models on the DeltaCenter.
                It will be used to group delta models.
            list_tags (:obj:`List`, *optional*, defaults to :obj:`None`): The tags in the form of list for the delta model, it is the
                optional identifiers that are not expected by `DeltaCenterArgument`. It will not be used to group delta models in the delta center
            dict_tags (:obj:`Dict`, *optional*, defaults to :obj:`None`): The tags in the form of dictionary for the delta model, it is the
                optional identifiers that are not expected by `DeltaCenterArgument`. It will not be used to group delta models in the delta center.
            delay_push (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether or not to delay the push to the DeltaCenter. When set to True,
                the delta object will be saved locally to save_directory, you can push it later using

            .. code-block:: shell

                python -m DeltaCenter upload save_directory


        """
        save_directory = finetuned_delta_path
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self.backbone_model# unwrap_model(self)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Save the config
        if save_config:
            if not hasattr(self, "config"):
                self.create_config_from_model()
            self.add_configs_when_saving()
            self.config.save_finetuned(save_directory)

        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

        final_center_args = self.create_delta_center_args(center_args=center_args,
                        center_args_pool=center_args_pool)

        state_dict_total_params = sum(p.numel() for p in state_dict.values())
        other_tags={}
        other_tags.update({'state_dict_total_params(M)':state_dict_total_params/1024/1024})
        other_tags.update({'test_result':test_result})
        if push_to_dc:
            self.create_yml(save_directory, final_center_args, list_tags, dict_tags,other_tags)

        if not delay_push:
            OssClient.upload(base_dir=save_directory)
        else:
            logger.info("\n"+"*"*30+f"\nYou delta models has been saved locally to:\n\t\t{os.path.abspath(save_directory)}\
                 \nyou can push it to the delta center later using \n\t\tpython -m DeltaCenter upload {os.path.abspath(save_directory)}\n"
                 +"*"*30)

        # get absolute path of saved_directory,


    def create_yml(self, save_dir, config, list_tags=None, dict_tags=None,other_tags=None):
        f = open("{}/config.yml".format(save_dir), 'w')
        config_dict = vars(config)
        config_dict['dict_tags'] = dict_tags if dict_tags is not None else {}
        config_dict['list_tags'] = list_tags if list_tags is not None else []
        if other_tags is not None:
            config_dict.update(other_tags)
        yaml.safe_dump(config_dict, f)
        f.close()

    @classmethod
    def from_finetuned(cls,
                       finetuned_delta_path: Optional[Union[str, os.PathLike]],
                       backbone_model: nn.Module,
                       delta_config = None,
                       cache_dir: Optional[Union[str, os.PathLike]] = None,
                       *model_args,
                       check_hash: Optional[bool] = True,
                       **kwargs):
        r"""
        Instantiate a finetuned delta model from a path.
        The backbone_model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated).
        To further train the model, you can use the :meth:`freeze_module <opendelta.basemodel.DeltaBase.freeze_module>` method.

        Parameters:

            finetuned_model_path (:obj:`str` or :obj:`os.PathLike`, *optional*):
                Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under a
                  user or organization name, like ``dbmdz/bert-base-german-cased``.
                - A path to a *directory* containing model weights saved using
                  :meth:`SaveLoadMixin.save_finetuned`, e.g., ``./my_model_directory/``.
                - A path or url to a *tensorflow index checkpoint file* (e.g, ``./tf_model/model.ckpt.index``). In
                  this case, ``from_tf`` should be set to ``True`` and a configuration object should be provided as
                  ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                  PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                  ``./flax_model/`` containing ``flax_model.msgpack``). In this case, ``from_flax`` should be set to
                  ``True``.
                - ``None`` if you are both providing the configuration and state dictionary (resp. with keyword
                  arguments ``config`` and ``state_dict``).
            backbone_model (:obj:`torch.nn.Module`): The backbone model to be modified.
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's ``__init__`` method.
            config (Union[:obj:`BaseDeltaConfig`, :obj:`str`, :obj:`os.PathLike`], *optional*): Can be either:
                - an instance of a class derived from :class:`~PretrainedConfig`,
                - a string or path valid as input to :py:meth:`~PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using :py:meth:`~PreTrainedModel.save_pretrained` and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (Dict[:obj:`str`, :obj:`torch.Tensor`], *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using :py:meth:`~PreTrainedModel.save_pretrained` and
                :py:meth:`~PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, *optional*, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, *optional*, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(:obj:`bool`, *optional*, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (:obj:`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token generated
                when running ``transformers-cli login`` (stored in ``~/.huggingface``).
            revision(:obj:`str`, *optional*, defaults to ``"main"``):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            mirror(:obj:`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            torch_dtype (:obj:`str` or :obj:`torch.dtype`, *optional*):
                Override the default :obj:`torch.dtype` and load the model under this dtype. If ``"auto"`` is passed the dtype
                will be automatically derived from the model's weights.

                .. warning::

                    This feature is inherited from HuggingFace. We do not guarantee its usefulness currently.
                    One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                    4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                    `pull request 11471 <https://github.com/huggingface/transformers/pull/11471>`_ for more information.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                ``output_attentions=True``). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:py:meth:`~PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's ``__init__`` function.

        .. tip::
            Passing ``use_auth_token=True`` is required when you want to use a private model.

        .. code-block:: python

            from transformers import AutoModelForSeq2SeqLM
            t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            from opendelta import AutoDeltaModel
            delta = AutoDeltaModel.from_finetuned("DeltaHub/lora_t5-base_mrpc", backbone_model=t5)
            delta.log()



        """
        # config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        # cache_dir = kwargs.pop("cache_dir", None)

        # ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        # output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        # _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        # low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(delta_config, BaseDeltaConfig):
            # config_path = delta_config if delta_config is not None else finetuned_model_path # Todo check
            delta_config, model_kwargs = cls.config_class.from_finetuned(
                finetuned_model_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )

        else:
            model_kwargs = kwargs

        print("delta_config", delta_config)
        # Load model
        if finetuned_model_path is not None:
            finetuned_model_path = str(finetuned_model_path)
            if os.path.isdir(finetuned_model_path):
                if os.path.isfile(os.path.join(finetuned_model_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(finetuned_model_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in "
                        f"directory {finetuned_model_path}."
                    )
            elif os.path.isfile(finetuned_model_path) or is_remote_url(finetuned_model_path):
                archive_file = finetuned_model_path
            else:
                archive_file = hf_bucket_url(
                    finetuned_model_path,
                    filename=WEIGHTS_NAME,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached #TODO

                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                )
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    f"Can't load weights for '{finetuned_model_path}'. Make sure that:\n\n"
                    )

                if revision is not None:
                    msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"

                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info(f"loading weights file {archive_file}")
            else:
                logger.info(f"loading weights file {archive_file} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # load pt weights early so that we know which dtype to init the model under

        if state_dict is None:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception as e:
                try:
                    with open(resolved_archive_file) as f:
                        if f.read().startswith("version"):
                            raise OSError(
                                "You seem to have cloned a repository without having git-lfs installed. Please install "
                                "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                                "you cloned."
                            )
                        else:
                            raise ValueError from e
                except (UnicodeDecodeError, ValueError):
                    raise OSError(
                        f"Unable to load weights from pytorch checkpoint file for '{finetuned_model_path}' "
                        f"at '{resolved_archive_file}'. "
                        "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
                    )

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry - we assume all weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
        dtype_orig = None
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    torch_dtype = next(iter(state_dict.values())).dtype
                else:
                    raise ValueError(
                        f"`torch_dtype` can be either a `torch.dtype` or `auto`, but received {torch_dtype}"
                    )
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)


        # Initialize the model from config and attach the delta model to the backbone_model.
        delta_model = cls.from_config(config, backbone_model, *model_args, **model_kwargs, )

        # load the state_dict into the backbone_model. As the delta model's parameter
        # is the same object as the deltas in the backbone model with different reference name,
        # the state_dict will also be loaded into the delta model.
        delta_model._load_state_dict_into_backbone(backbone_model, state_dict)

        backbone_hash = gen_model_hash(backbone_model)
        if check_hash and hasattr(config, "backbone_hash") and \
                          config.backbone_hash is not None and \
                          config.backbone_hash != backbone_hash:
            logger.warning("The config has an hash of the backbone model, and is"
                            "different from the hash of the loaded model. This indicates a mismatch"
                            "between the backbone model that the delta checkpoint is based on and"
                            "the one you loaded. You propobability need to Train the model instead of"
                            "directly inference. ")

        # Set model in evaluation mode to deactivate DropOut modules by default
        backbone_model.eval()

        return delta_model


    def create_delta_center_args(self, center_args, center_args_pool):
        """
        Create the delta center args for the center model.
        center_args has higher priority than center_args_pool.

        """
        mdict = {}
        field = fields(DeltaCenterArguments)


        for f in field:
            exist = False
            # first is center_args, exact match
            if f.name in center_args:
                mdict[f.name] = center_args[f.name]
                continue
            # second is center_args_pool, can use alternative names
            if f.name in center_args_pool:
                mdict[f.name] = center_args_pool[f.name]
                exist = True
            elif f.name in alternative_names:
                for altername in alternative_names[f.name]:
                    if altername in center_args_pool:
                        mdict[f.name] = center_args_pool[altername]
                        exist = True
                        break
            # if not exist, find from self.stat or set to default
            if not exist:
                if f.name in self.stat:
                    mdict[f.name] = self.stat[f.name]
                else:
                    mdict[f.name] = f.default

        # if eventualy name is not set, create a default one
        if mdict['name'] is None or mdict['name'] == '':
            print("Warning: name is not set, use default name")
            mdict['name'] = self.create_default_name(**mdict)


        center_args = DeltaCenterArguments(**mdict)
        return  center_args

    def create_default_name(self, **kwargs):
        r"""Currently, it's only a simple concatenation of the arguments.
        """
        print("key args", kwargs)

        reponame = ""
        reponame += kwargs["model_path_public"].split("/")[-1]+"_" if kwargs['model_path_public'] is not None else kwargs['backbone_model']
        reponame += kwargs["delta_type"]+"_" if kwargs["delta_type"] is not None else ""

        # tasks
        if isinstance(kwargs["train_tasks"], list):
            train_tasks = "+".join(kwargs["train_tasks"])
        elif kwargs["train_tasks"] is not None:
            train_tasks = kwargs["train_tasks"]
        else:
            logger.warning("train_tasks are not find in all arguments. Do you miss them?")
            train_tasks = None
        reponame += train_tasks+"_" if train_tasks is not None else ""

        # time
        reponame += datetime.datetime.now().strftime("%Y%m%d%H%M%S") #+ gen_model_hash(model=self.backbone_model)

        # model hash
        if hasattr(self.config, "backbone_hash"):
            reponame += self.config.backbone_hash[:3]
        return reponame

