import os
import bs4
import random
import getpass

from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OPENAI_API_KEY"] = getpass.getpass()
llm = ChatOpenAI(model="gpt-4o-mini")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

candidate1 = vectorstore.as_retriever()
# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
candidate2 = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)
# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
candidate3 = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)
# Only retrieve documents that have a relevance score
# Above a certain threshold
candidate4 = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)
# Only get the single most similar document from the dataset
candidate5 = vectorstore.as_retriever(search_kwargs={'k': 1})
# Use a filter to only retrieve documents from a specific paper
candidate6 = vectorstore.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
vectorstore_candidates = [candidate1, candidate2, candidate3, candidate4, candidate5, candidate6]

user_query = "agent memory"

parser = JsonOutputParser()

retriever = random.choice(vectorstore_candidates)
retrieval_result = retriever.invoke(user_query)
template = "retrieval result is {retrieval_result}. Please evaluate the relevance between retrieval result and {user_query}. \nReturn a JSON object with key 'relevance' and answer should be yes or no\n"

prompt = PromptTemplate(
    template=template,
    input_variables=["retrieval_result", "user_query"]
)

chain = prompt | llm | parser

res = chain.invoke({"retrieval_result": retrieval_result, "user_query": user_query})

if res["relevance"] == "no":
    print("The retrieval result is not relevant.")

template = "retrieval result is {retrieval_result} and user query is {user_query}. Please find that there is a hallucination in the retrieval result. \nReturn a JSON object with key 'hallucination' and answer should be yes or no\n"

prompt = PromptTemplate(
        template=template,
        input_variables=["retrieval_result", "user_query"]
    )

for i in range(3):
    chain = prompt | llm | parser

    res = chain.invoke({"retrieval_result": retrieval_result, "user_query": user_query})

    if res["hallucination"] == "no":
        break

    retriever = random.choice(vectorstore_candidates)
    retrieval_result = retriever.invoke(user_query)

for document in retrieval_result:
    print(document.page_content)
