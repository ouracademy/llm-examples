from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.callbacks import get_openai_callback

import os
from dotenv import load_dotenv
import json
import tiktoken


load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain_1.db"))


def get_model(model_id="gpt-3.5-turbo", apikey=os.getenv("gpt-3.5-turbo"), config={"temperature": 0}):
    llm_openai = ChatOpenAI(openai_api_key= apikey, **config)
    mapping = {
        "gpt-3.5-turbo": llm_openai,
    }
    return mapping.get(model_id, None)

def build_template(template_id, template, payload):
    result = template.format(**payload)
    return result

class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        text = text.replace("\n", "")
        text = text.replace('""', '","')
        idx = text.index("{")
        if idx:
            text = text[idx:]
        try:
            y = json.loads(text)
            z = json.dumps(y)
        except ValueError:
            return text
        return z

def get_chain(prompt_template, model):
    return LLMChain(
        llm=model,
        prompt=prompt_template,
        output_parser=JsonOutputParser(),
        verbose=True,
    )
    # prompt_2 | llm_1,
    # prompt_1 | llm_2 | JsonOutputParser(),
    # prompt_2 | llm_2

def run_chain(chain, payload_prompt, request_id, tags=[]):
    prompt_input = {**payload_prompt}
    result = chain.invoke(prompt_input, {"tags": tags, "run_name": request_id})
    return result

def num_tokens(model_id, text):
    encoding = tiktoken.encoding_for_model(model_id)
    num_tokens = len(encoding.encode(text))
    return num_tokens

def get_cost(num_tokens_input, num_tokens_output, model):
    mapping = {
        "gpt-3.5-turbo": [0.0010, 0.0020],
        "gpt-4": [0.03, 0.06],
        "gpt-3.5-turbo-instruct": [0.0015, 0.0020],
    }
    cost = mapping[model]
    input = cost[0] * num_tokens_input / 1000
    output = cost[1] * num_tokens_output / 1000
    input = round(input, 4)
    output = round(output, 4)
    return input, output

def add_metadata(result, cb, model_id):
    if cb.prompt_tokens > 0:
        result["cb_tokens_input"] = cb.prompt_tokens
        result["cb_tokens_output"] = cb.completion_tokens
        result["cb_price_total"] = cb.total_cost

    output = result["text"]
    input = result["input"]
    model_tiktoken = model_id
    num_tokens_input = num_tokens(model_tiktoken, input)
    num_tokens_output = num_tokens(model_tiktoken, output)
    costo_input, costo_output = get_cost(
        num_tokens_input, num_tokens_output, model_tiktoken
    )
    result["tokens_input"] = num_tokens_input
    result["tokens_output"] = num_tokens_output
    result["price_output"] = costo_output
    result["price_input"] = costo_input
    result["price_total"] = costo_input + costo_output
    return result

def invoque_llm_messages(messages, input,  model_id="gpt-3.5-turbo", config={}, task="", id=""):
    #prompt = PromptTemplate.from_template("{input}")
    messages_formated = [ ( i["role"], i["content"]) for i in messages ]
    chat_prompt = ChatPromptTemplate.from_messages(messages_formated)
    chain = LLMChain(
        llm=get_model(model_id, config=config),
        prompt= chat_prompt,
        verbose=True,
    )
    with get_openai_callback() as cb:
        result = chain.invoke({"input": input}, {"tags": [], "run_name": f"{id}__{task}"})
        result = add_metadata(result, cb, model_id)
        result["id_persona"] = id
        result["task"] = task
    return result

def invoque_llm(input,  model, task="", id=""):
    prompt = PromptTemplate.from_template("{input}")

    chain = LLMChain(
        llm=model,
        prompt= prompt,
        verbose=True,
    )
    with get_openai_callback() as cb:
        result = chain.invoke({"input": input}, {"tags": [], "run_name": f"{id}__{task}"})
        result = add_metadata(result, cb, model.model_name)
        result["id_persona"] = id
        result["task"] = task
    return result
