import getpass
import os
import dotenv
import redis

dotenv.load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda

from retrieve import vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class RagAgent:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "This is a document from the food insecurity of california. Retrieved the {context} from the document"),
                #MessagesPlaceholder("chat_history"),
                ("human", "Answer the following {question} using the above retrieved documents")
            ]
        )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
        )
        self.retriever = vector_store.as_retriever()
        self.chain = {"context": self.retriever | format_docs, "question": RunnablePassthrough()}| self.prompt| self.llm | StrOutputParser()
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        self.chat_retriever_chain = create_history_aware_retriever(
            llm, retriever, rephrase_prompt
        )

    

CHAIN = RagAgent().chain

class Redis_Client:
    def __init__():
        self.client = redis.Redis(host='localhost', port=6379, db=0)

    def _search_in_redis(self, query:str):
        answer = self.client.get(query)
        return answer or CHAIN.invoke(query)


queries = ['List major Food insecurity reason in 2024','Explain malnutrition in war zones','Explain increase prices impact on food security']
# get_query(queries)
redis_client = get_redis()
print("\n\n",redis_client.get(queries[2]))