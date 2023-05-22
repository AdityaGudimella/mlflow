"""
The ``mlflow.semantic_kernel`` module provides an API for logging and loading SemanticKernel models.
This module exports multivariate LangChain models in the semantic_kernel flavor and univariate
LangChain models in the pyfunc flavor:

LangChain (native) format
    This is the main flavor that can be accessed with LangChain APIs.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and for batch inference.

.. _LangChain:
    https://github.com/microsoft/semantic-kernel/blob/main/python/README.md
"""
import logging
import os
import typing as t
from pathlib import Path

import pandas as pd
import cloudpickle
import json
import numpy as np
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_OPENAI_TESTING
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.openai.utils import TEST_CONTENT

logger = logging.getLogger(mlflow.__name__)

FLAVOR_NAME = "semantic_kernel"
_MODEL_DATA_FILE_NAME = "model.yaml"
_MODEL_DATA_KEY = "model_data"
_AGENT_PRIMITIVES_FILE_NAME = "agent_primitive_args.json"
_AGENT_PRIMITIVES_DATA_KEY = "agent_primitive_data"
_AGENT_DATA_FILE_NAME = "agent.yaml"
_AGENT_DATA_KEY = "agent_data"
_TOOLS_DATA_FILE_NAME = "tools.pkl"
_TOOLS_DATA_KEY = "tools_data"
_MODEL_TYPE_KEY = "model_type"


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("semantic_kernel")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    sk_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a LangChain model to a path on the local file system.

    :param sk_model: An LLMChain model.
    :param path: Local path where the serialized model (as YAML) is to be saved.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `sk_model.input_keys` and `sk_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    """
    import semantic_kernel as sk

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    else:
        input_signature: t.List[t.Tuple[str, np.dtype[t.Any]]] = []
        input_schema = Schema(
            [
                ColSpec(type=DataType.from_numpy_dtype(dtype), name=name)
                for name, dtype in input_signature
            ]
        )
        output_signature: t.List[t.Tuple[str, np.dtype[t.Any]]] = []
        output_schema = Schema(
            [
                ColSpec(type=DataType.from_numpy_dtype(dtype), name=name)
                for name, dtype in output_signature
            ]
        )
        mlflow_model.signature = ModelSignature(input_schema, output_schema)

    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_kwargs = _save_model(sk_model, path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.semantic_kernel",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        **model_data_kwargs,
    )
    flavor_conf = {
        _MODEL_TYPE_KEY: sk_model.__class__.__name__,
        **model_data_kwargs,
    }
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        semantic_kernel_version=sk.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                str(path), FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    sk_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a LangChain model as an MLflow artifact for the current run.

    :param sk_model: LangChain model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.

    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output
                      :py:class:`Schema <mlflow.types.Schema>`.
                      If not specified, the model signature would be set according to
                      `sk_model.input_keys` and `sk_model.output_keys` as columns names, and
                      `DataType.string` as the column type.
                      Alternatively, you can explicitly specify the model signature.
                      The model signature can be :py:func:`inferred
                      <mlflow.models.infer_signature>` from datasets with valid model input
                      (e.g. the training dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training dataset),
                      for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        chain = LLMChain(llm=llm, prompt=prompt)
                        prediction = chain.run(input_str)
                        input_columns = [
                            {"type": "string", "name": input_key} for input_key in chain.input_keys
                        ]
                        signature = infer_signature(input_columns, predictions)

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to
                          feed the model. The given example will be converted to a
                          Pandas DataFrame and then serialized to json using the
                          Pandas split-oriented format. Bytes are base64-encoded.

    :param await_registration_for: Number of seconds to wait for the model version
                        to finish being created and is in ``READY`` status.
                        By default, the function waits for five minutes.
                        Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    import semantic_kernel

    if not isinstance(
        sk_model, (langchain.chains.llm.LLMChain, langchain.agents.agent.AgentExecutor)
    ):
        raise mlflow.MlflowException.invalid_parameter_value(
            "MLflow langchain flavor only supports logging langchain.chains.llm.LLMChain and "
            + f"langchain.agents.agent.AgentExecutor instances, found {type(sk_model)}"
        )

    _SUPPORTED_LLMS = {langchain.llms.openai.OpenAI, langchain.llms.huggingface_hub.HuggingFaceHub}
    if (
        isinstance(sk_model, langchain.chains.llm.LLMChain)
        and type(sk_model.llm) not in _SUPPORTED_LLMS
    ):
        logger.warning(
            "MLflow does not guarantee support for LLMChains outside of HuggingFaceHub and "
            "OpenAI, found %s",
            type(sk_model.llm).__name__,
        )

    if (
        isinstance(sk_model, langchain.agents.agent.AgentExecutor)
        and type(sk_model.agent.llm_chain.llm) not in _SUPPORTED_LLMS
    ):
        logger.warning(
            "MLflow does not guarantee support for LLMChains outside of HuggingFaceHub and "
            "OpenAI, found %s",
            type(sk_model.agent.llm_chain.llm).__name__,
        )

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.semantic_kernel,
        registered_model_name=registered_model_name,
        sk_model=sk_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


def _save_model(model, path):
    """Save a SemanticKernel Kernel to a path.

    Args:
        model (semantic_kernel.Kernel): The model to save.
        path (str): The path to which to save the model.

    Raises:
        MlflowException: If the model is not a SemanticKernel Kernel.
    """
    import semantic_kernel as sk

    model_data_path = Path(path) / _MODEL_DATA_FILE_NAME
    model_data_kwargs = {_MODEL_DATA_KEY: _MODEL_DATA_FILE_NAME}

    try:
        model_data_path.write_bytes(sk.to_json(model))
    except (TypeError, NotImplementedError) as e:
        raise mlflow.MlflowException.invalid_parameter_value(
            "MLflow SemanticKernel flavor only supports saving Kernel "
            + f"instances, found {type(model)}"
        ) from e
    return model_data_kwargs

class _LangChainModelWrapper:
    def __init__(self, sk_model):
        self.sk_model = sk_model

    def predict(self, data: t.Union[pd.DataFrame, t.List[t.Union[str, t.Dict[str, str]]]]) -> t.List[str]:
        from mlflow.langchain.api_request_parallel_processor import process_api_requests

        if isinstance(data, pd.DataFrame):
            messages = data.to_dict(orient="records")
        elif isinstance(data, list) and (
            all(isinstance(d, str) for d in data) or all(isinstance(d, dict) for d in data)
        ):
            messages = data
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                "Input must be a pandas DataFrame or a list of strings or a list of dictionaries",
            )
        return process_api_requests(sk_model=self.sk_model, requests=messages)


class _TestLangChainWrapper(_LangChainModelWrapper):
    """
    A wrapper class that should be used for testing purposes only.
    """

    def predict(self, data):
        import semantic_kernel
        from tests.semantic_kernel.test_semantic_kernel_model_export import _mock_async_request

        if isinstance(self.sk_model, langchain.chains.llm.LLMChain):
            mockContent = TEST_CONTENT
        elif isinstance(self.sk_model, langchain.agents.agent.AgentExecutor):
            mockContent = f"Final Answer: {TEST_CONTENT}"

        with _mock_async_request(mockContent):
            return super().predict(data)


def _load_pyfunc(path):
    """
    Load PyFunc implementation for LangChain. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``semantic_kernel`` flavor.
    """
    wrapper_cls = _TestLangChainWrapper if _MLFLOW_OPENAI_TESTING.get() else _LangChainModelWrapper
    return wrapper_cls(_load_model_from_local_fs(path))


def _load_model_from_local_fs(local_model_path: str):
    """Load a SemanticKernel Kernel from a path on the local file system.

    Args:
        local_model_path (str): The path from which to load the model.

    Returns:
        semantic_kernel.Kernel: The loaded model.
    """
    import semantic_kernel as sk
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sk_model_path = os.path.join(
        local_model_path, flavor_conf.get(_MODEL_DATA_KEY, _MODEL_DATA_FILE_NAME)
    )
    return sk.from_json(Path(sk_model_path).read_bytes())


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load a LangChain model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A LangChain model instance
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    return _load_model_from_local_fs(local_model_path)
