import os
import bs4
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# Retrieve and generate using the relevant snippets of the blog.
user_query = "agent memory"
retrieval_result = list()

retriever = vectorstore.as_retriever()
retrieval_result.append(retriever.invoke(user_query))

# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)
retrieval_result.append(retriever.invoke(user_query))

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)
retrieval_result.append(retriever.invoke(user_query))

# Only retrieve documents that have a relevance score
# Above a certain threshold
vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)
retrieval_result.append(retriever.invoke(user_query))

# Only get the single most similar document from the dataset
vectorstore.as_retriever(search_kwargs={'k': 1})
retrieval_result.append(retriever.invoke(user_query))

# Use a filter to only retrieve documents from a specific paper
vectorstore.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
retrieval_result.append(retriever.invoke(user_query))

query = ("Please evaluate the retrieval quality. If it is relevant, please provide a result as Json format {"
         "'relevance':'yes'}. If it is not relevant, please provide a result as Json format {'relevance':'no'}.")

parser = JsonOutputParser()
template = "retrieval result is {retrieval_result}. Please evaluate the relevance between retrieval result and {user_query}. \nReturn a JSON object with key 'relevance' and answer should be yes or no\n"

prompt = PromptTemplate(
    template=template,
    input_variables=["retrieval_result", "user_query"]
)

chain = prompt | llm | parser

for result in retrieval_result:
    res = chain.invoke({"retrieval_result": result, "user_query": user_query})
    print(res)

# retrieval_result.clear()
#
# user_query = "I like an apple."
# retriever = vectorstore.as_retriever()
# retrieval_result.append(retriever.invoke(user_query))
#
# # Retrieve more documents with higher diversity
# # Useful if your dataset has many similar documents
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 6, 'lambda_mult': 0.25}
# )
# retrieval_result.append(retriever.invoke(user_query))
#
# # Fetch more documents for the MMR algorithm to consider
# # But only return the top 5
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 5, 'fetch_k': 50}
# )
# retrieval_result.append(retriever.invoke(user_query))
#
# # Only retrieve documents that have a relevance score
# # Above a certain threshold
# vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.8}
# )
# retrieval_result.append(retriever.invoke(user_query))
#
# # Only get the single most similar document from the dataset
# vectorstore.as_retriever(search_kwargs={'k': 1})
# retrieval_result.append(retriever.invoke(user_query))
#
# # Use a filter to only retrieve documents from a specific paper
# vectorstore.as_retriever(
#     search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
# )
# retrieval_result.append(retriever.invoke(user_query))
#
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["retrieval_result", "user_query"]
# )
#
# chain = prompt | llm | parser
#
# for result in retrieval_result:
#     res = chain.invoke({"retrieval_result": result, "user_query": user_query})
#     print(res)

template = "retrieval result is {retrieval_result} and user query is {user_query}. Please find that there is a hallucination in the retrieval result. \nReturn a JSON object with key 'hallucination' and answer should be yes or no\n"
prompt = PromptTemplate(
    template=template,
    input_variables=["retrieval_result", "user_query"]
)

chain = prompt | llm | parser

for result in retrieval_result:
    res = chain.invoke({"retrieval_result": result, "user_query": user_query})
    print(res)
