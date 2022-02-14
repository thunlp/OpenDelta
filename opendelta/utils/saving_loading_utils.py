
from io import RawIOBase
from tarfile import HeaderError
from typing import Union, Optional, Callable
from opendelta.delta_configs import BaseDeltaConfig
from opendelta.utils.model_md5 import gen_model_hash
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

logger = logging.get_logger(__name__)

class SaveLoadMixin(PushToHubMixin):
    def add_configs_when_saving(self,):
        self.config.backbone_class = self.backbone_model.__class__.__name__
        self.config.backbone_checkpoint_name = os.path.split(self.backbone_model.config._name_or_path.strip("/"))[-1]
        self.config.backbone_hash = gen_model_hash(self.backbone_model)




    def save_finetuned(
        self,
        save_directory: Optional[Union[str, os.PathLike]] = "./output/",
        save_config: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
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
            push_to_hub (:obj:`bool`, *optional*, defaults to :obj:`False`):
                Whether or not to push your model to the HuggingFace model hub after saving it.
                
                .. tip::

                    Using ``push_to_hub=True`` will synchronize the repository you are pushing to with ``save_directory``,
                    which requires ``save_directory`` to be a local clone of the repo you are pushing to if it's an existing
                    folder. Pass along ``temp_dir=True`` to use a temporary directory instead.
                
            kwargs:
                Additional key word arguments passed along to the :py:meth:`~file_utils.PushToHubMixin.push_to_hub` method.

        .. note::

            You may need to install git-lfs on your machine. 
        
            .. code-block:: bash
            
                wget -P ~ https://github.com/git-lfs/git-lfs/releases/download/v3.0.2/git-lfs-linux-amd64-v3.0.2.tar.gz
                cd ~
                tar -xvzf git-lfs-linux-amd64-v3.0.2.tar.gz
                export PATH=~:$PATH
                git-lfs install

        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
    
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

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    @classmethod
    def from_finetuned(cls, 
                        finetuned_model_name_or_path: Optional[Union[str, os.PathLike]], 
                        backbone_model: nn.Module,
                        *model_args,
                        check_hash: Optional[bool] = True,
                        **kwargs):
        r"""
        Instantiate a finetuned delta model from a path.
        The backbone_model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). 
        To further train the model, you can use the :meth:`freeze_module <opendelta.basemodel.DeltaBase.freeze_module>` method.

        Parameters:

            finetuned_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, *optional*): 
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
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        
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
        if not isinstance(config, BaseDeltaConfig):
            config_path = config if config is not None else finetuned_model_name_or_path
            config, model_kwargs = cls.config_class.from_finetuned(
                config_path,
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

        # Load model
        if finetuned_model_name_or_path is not None:
            finetuned_model_name_or_path = str(finetuned_model_name_or_path)
            if os.path.isdir(finetuned_model_name_or_path):
                if os.path.isfile(os.path.join(finetuned_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(finetuned_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {WEIGHTS_NAME} found in "
                        f"directory {finetuned_model_name_or_path}."
                    )
            elif os.path.isfile(finetuned_model_name_or_path) or is_remote_url(finetuned_model_name_or_path):
                archive_file = finetuned_model_name_or_path
            else:
                archive_file = hf_bucket_url(
                    finetuned_model_name_or_path,
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
                    f"Can't load weights for '{finetuned_model_name_or_path}'. Make sure that:\n\n"
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
                        f"Unable to load weights from pytorch checkpoint file for '{finetuned_model_name_or_path}' "
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
    
