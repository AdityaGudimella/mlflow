import typing as t
from contextlib import contextmanager
from pathlib import Path

import semantic_kernel as sk
import mlflow
import pytest
import transformers
import json
from pyspark.sql import SparkSession

from tests.helper_functions import pyfunc_serve_and_score_model
from mlflow.exceptions import MlflowException
from mlflow.openai.utils import (
    _mock_chat_completion_response,
    _mock_request,
    _MockResponse,
    TEST_CONTENT,
)


@contextmanager
def _mock_async_request(content=TEST_CONTENT):
    with _mock_request(return_value=_mock_chat_completion_response(content)) as m:
        yield m


@pytest.fixture
def model_path(tmp_path):
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def spark():
    with SparkSession.builder.master("local[*]").getOrCreate() as s:
        yield s


def create_huggingface_model(model_path):
    architecture = "lordtt13/emo-mobilebert"
    mlflow.transformers.save_model(
        transformers_model={
            "model": transformers.TFMobileBertForSequenceClassification.from_pretrained(
                architecture
            ),
            "tokenizer": transformers.AutoTokenizer.from_pretrained(architecture),
        },
        path=model_path,
    )
    llm = mlflow.transformers.load_model(model_path)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    hf_pipe = HuggingFacePipeline(pipeline=llm)
    return LLMChain(llm=hf_pipe, prompt=prompt)


def create_openai_llmchain():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    return LLMChain(llm=llm, prompt=prompt)


def create_openai_llmagent():
    from langchain.agents import load_tools
    from langchain.agents import initialize_agent
    from langchain.agents import AgentType

    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(temperature=0)

    # Next, let's load some tools to use.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # Finally, let's initialize an agent with the tools.
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


def create_model(llm_type, model_path=None):
    if llm_type == "openai":
        return create_openai_llmchain()
    if llm_type == "huggingfacehub":
        return create_huggingface_model(model_path)
    if llm_type == "openaiagent":
        return create_openai_llmagent()
    if llm_type == "fake":
        return FakeLLM()
    raise NotImplementedError("This model is not supported yet.")


# class FakeLLM(LLM):
#     """Fake LLM wrapper for testing purposes."""

#     queries: Optional[Mapping] = None

#     @property
#     def _llm_type(self) -> str:
#         """Return type of llm."""
#         return "fake"

#     # pylint: disable=arguments-differ
#     def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None) -> str:
#         """First try to lookup in queries, else return 'foo' or 'bar'."""
#         if self.queries is not None:
#             return self.queries[prompt]
#         if stop is None:
#             return "foo"
#         else:
#             return "bar"

#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {}


# class FakeChain(Chain):
#     """Fake chain class for testing purposes."""

#     be_correct: bool = True
#     the_input_keys: List[str] = ["foo"]
#     the_output_keys: List[str] = ["bar"]

#     @property
#     def input_keys(self) -> List[str]:
#         """Input keys."""
#         return self.the_input_keys

#     @property
#     def output_keys(self) -> List[str]:
#         """Output key of bar."""
#         return self.the_output_keys

#     # pylint: disable=arguments-differ
#     def _call(self, inputs: Dict[str, str], run_manager=None) -> Dict[str, str]:
#         if self.be_correct:
#             return {"bar": "baz"}
#         else:
#             return {"baz": "bar"}


@pytest.mark.skip()
def test_langchain_native_log_and_load_model():
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string]"
    assert str(logged_model.signature.outputs) == "['text': string]"

    assert type(loaded_model) == langchain.chains.llm.LLMChain
    assert type(loaded_model.llm) == langchain.llms.openai.OpenAI
    assert type(loaded_model.prompt) == langchain.prompts.PromptTemplate
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"


@pytest.mark.skip()
def test_pyfunc_load_openai_model():
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")

    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

    assert "langchain" in logged_model.flavors
    assert type(loaded_model) == mlflow.pyfunc.PyFuncModel


@pytest.mark.skip()
def test_langchain_model_predict():
    with _mock_request(return_value=_mock_chat_completion_response()):
        model = create_model("openai")
        with mlflow.start_run():
            logged_model = mlflow.langchain.log_model(model, "langchain_model")
        loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
        result = loaded_model.predict([{"product": "MLflow"}])
        assert result == [TEST_CONTENT]


@pytest.mark.skip()
def test_pyfunc_spark_udf_with_langchain_model(spark):
    model = create_model("openai")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    loaded_model = mlflow.pyfunc.spark_udf(spark, logged_model.model_uri, result_type="string")
    df = spark.createDataFrame([("MLflow",), ("Spark",)], ["product"])
    df = df.withColumn("answer", loaded_model())
    pdf = df.toPandas()
    assert pdf["answer"].tolist() == [TEST_CONTENT, TEST_CONTENT]


@pytest.mark.skip()
def test_langchain_agent_model_predict():
    langchain_agent_output = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "text": f"Final Answer: {TEST_CONTENT}",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    model = create_model("openaiagent")
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "langchain_model")
    loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    langchain_input = {
        "input": "What was the high temperature in SF yesterday in Fahrenheit? "
        "What is that number raised to the .023 power?"
    }
    with _mock_request(return_value=_MockResponse(200, langchain_agent_output)):
        result = loaded_model.predict([langchain_input])
        assert result == [TEST_CONTENT]

    inference_payload = json.dumps({"inputs": langchain_input})
    langchain_agent_output_serving = {"predictions": langchain_agent_output}
    with _mock_request(return_value=_MockResponse(200, langchain_agent_output_serving)):
        import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
        from mlflow.deployments import PredictionsResponse

        response = pyfunc_serve_and_score_model(
            logged_model.model_uri,
            data=inference_payload,
            content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
            extra_args=["--env-manager", "local"],
        )

        assert (
            PredictionsResponse.from_json(response.content.decode("utf-8"))
            == langchain_agent_output_serving
        )


@pytest.mark.skip()
def test_unsupported_chain_types():
    chain = FakeChain()
    with pytest.raises(
        MlflowException,
        match="MLflow langchain flavor only supports logging langchain.chains.llm.LLMChain",
    ):
        with mlflow.start_run():
            mlflow.semantic_kernel.log_model(chain, "fake_chain_model")


@pytest.fixture()
def model():
    """SemanticKernel wrapper model."""
    return sk.Kernel()


def test_native_log_and_load_model(model):
    """I should be able to save and load SK models using mlflow."""
    with mlflow.start_run():
        logged_model = mlflow.semantic_kernel.log_model(
            model, "temp_sk_model", registered_model_name="SemanticKernelModel"
        )

    loaded_model = mlflow.semantic_kernel.load_model(logged_model.model_uri)
    assert "semantic_kernel" in logged_model.flavors
    assert isinstance(loaded_model, type(model))
    assert isinstance(loaded_model.memory, type(model.memory))
    assert isinstance(loaded_model.prompt_template_engine, type(model.prompt_template_engine))
    assert isinstance(loaded_model.skills, type(model.skills))


def test_native_save_and_load_model(model, model_path: Path):
    """I should be able to save and load SK models using mlflow."""
    mlflow.semantic_kernel.save_model(model, model_path)

    loaded_model = mlflow.semantic_kernel.load_model(model_path)
    assert isinstance(loaded_model, type(model))
    assert isinstance(loaded_model.memory, type(model.memory))
    assert isinstance(loaded_model.prompt_template_engine, type(model.prompt_template_engine))
    assert isinstance(loaded_model.skills, type(model.skills))


def test_langchain_native_log_and_load_model(model):
    with mlflow.start_run():
        logged_model = mlflow.langchain.log_model(model, "sk_model")

    loaded_model = mlflow.langchain.load_model(logged_model.model_uri)

    assert "semantic_kernel" in logged_model.flavors
    assert str(logged_model.signature.inputs) == "['product': string]"
    assert str(logged_model.signature.outputs) == "['text': string]"

    assert isinstance(loaded_model, type(model))
    assert isinstance(loaded_model.memory, type(model.memory))
    assert isinstance(loaded_model.prompt_template_engine, type(model.prompt_template_engine))
    assert isinstance(loaded_model.skills, type(model.skills))
    assert loaded_model.prompt.template == "What is a good name for a company that makes {product}?"
