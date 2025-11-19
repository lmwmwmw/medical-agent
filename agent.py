from langchain_chroma import Chroma
from langchain_classic.chains import LLMChain
from langchain_core.prompts.prompt import PromptTemplate
from prompt import *
from utils import *
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
class Agent():
    def __init__(self):
        self.vdb = Chroma(
            persist_directory=os.path.join(os.path.dirname(__file__), './data/db/'),
            embedding_function=get_embeddings_model()
        )

    #常规情况
    def generic(self,query):
        prompt = PromptTemplate.from_template(GENERIC_PROMPT_TPL)
        llm_chain = (prompt | get_llm_model()).with_config({"verbose":"VERBOSE"})
        return llm_chain.invoke(query)

    # 调用知识库
    def retrieval(self,query):
        # 召回并过滤文档
        documents = self.vdb.similarity_search_with_relevance_scores(query,k=5)

        query_result = [doc[0].page_content for doc in documents if doc[1]>0.7]
        #书写提示词
        prompt = PromptTemplate.from_template(RETRIVAL_PROMPT_TPL)

        chain = (prompt | get_llm_model()).with_config({"verbose":"VERBOSE"})
        input = {
            'query': query,
            'query_result': '\n\n'.join(query_result) if len(query_result)  else '没有查到',
        }
        return chain.invoke(input)

    def graph_func(self,query):
        # 命名实体识别
        response_schemas = [
            ResponseSchema(type='list', name='disease', description='疾病名称实体'),
            ResponseSchema(type='list', name='symptom', description='疾病症状实体'),
            ResponseSchema(type='list', name='drug', description='药品名称实体'),
        ]

        format_instructions = structured_output_parser(response_schemas)

        prompt = PromptTemplate(
            template=NER_PROMPT_TPL,
            partial_variables={'format_instructions':format_instructions},
            input_variables=['query']
        )
        # 交给llm提取实体
        chain = (prompt | get_llm_model()).with_config({"verbose":"VERBOSE"})
        response = chain.invoke(query)

        # 输出解析为json格式
        output_parser = StructuredOutputParser(response_schemas=response_schemas)
        output_parser.parse(response)



if __name__ == '__main__':
    agent = Agent()
    # print(agent.generic('你是谁？'))
    # print(agent.retrieval('介绍一下寻医问药网？'))
    # print(agent.retrieval('寻医问药网的客服是多少？'))
    agent.graph_func()