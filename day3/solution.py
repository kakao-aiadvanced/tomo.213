import os

from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from tavily import TavilyClient

tavily = TavilyClient(api_key='api key')

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

### Retrieval Grader

system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Generate

system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Hallucination Grader

system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)

hallucination_grader = prompt | llm | JsonOutputParser()

### Answer Grader


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)


# Prompt
system = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)

answer_grader = prompt | llm | JsonOutputParser()

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    web_search_tried: bool
    sources: List[str]
    stop: bool


### Nodes

# Docs Retrieval
def docs_retrieval(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    print(question)
    # print(documents)
    return {"documents": documents, "question": question}

# Relevance Checker
def relevance_checker(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    web_search_tried = state["web_search_tried"] if "web_search_tried" in state else False
    sources = state["sources"] if "sources" in state else []

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            if "source" in d.metadata:
                sources.append(d.metadata["source"])
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            continue

    if (len(filtered_docs) == 0) and (web_search_tried == True):
        print("---NO RELEVANT DOCUMENTS, STOP CHAIN---")
        return {"stop": True}

    return {"documents": filtered_docs, "question": question, "sources": sources}

# Answer Generation
def answer_generator(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    sources = state["sources"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "sources":sources}

# Hallucination Checker
def hallucination_checker(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    sources = state["sources"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return {"question":question, "documents": documents, "generation":generation, "hallucinated": "useful", "sources": sources}
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return {"question":question, "documents": documents, "generation":generation, "hallucinated": "not useful", "sources": sources}
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return {"question":question, "documents": documents, "generation":generation, "hallucinated": "not supported"}

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    sources = state["sources"]

    # Web search
    docs = tavily.search(query=question)['results']

    web_results = "\n".join([d["content"] for d in docs])
    sources.append([d['url'] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = web_results
    return {"documents": documents, "question": question, "web_search_tried": True, "sources": sources}


### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    documents = state["documents"]
    stop = state["stop"] if "stop" in state else False

    if stop:
        return "stop"

    if len(documents) == 0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def decide_to_answer(state):
    """
        Determines whether to generate an answer, or retry

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

    print("---ASSESS GRADED DOCUMENTS---")
    hallucinated = state["hallucinated"]

    if hallucinated == "not supported":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "retry"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "finished"

### Conditional edge


# Set up the graph
os.environ['OPENAI_API_KEY'] = 'api key'

llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("docs_retrieval", docs_retrieval)  # docs_retrieval
workflow.add_node("relevance_checker", relevance_checker)  # relevance_checker
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("answer_generator", answer_generator)  # answer_generator
workflow.add_node("hallucination_checker", hallucination_checker) # hallucination_checker

# Build graph
workflow.set_entry_point("docs_retrieval")
workflow.add_edge("docs_retrieval", "relevance_checker")
workflow.add_conditional_edges(
    "relevance_checker",
    decide_to_generate,
    {
        "stop": END,
        "websearch": "websearch",
        "generate": "answer_generator",
    },
)
workflow.add_edge("websearch", "relevance_checker")
workflow.add_edge("answer_generator", "hallucination_checker")
workflow.add_conditional_edges(
    "hallucination_checker",
    decide_to_answer,
    {
        "retry": "answer_generator",
        "finished": END
    },
)

# Compile
# app = workflow.compile()
# inputs = {"question": "What is prompt engineering?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(f"Result: {value['generation']}")
# pprint(f"Sources: {value['sources']}")

# Compile
app = workflow.compile()
inputs = {"question": "Where does Messi play right now?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(f"Result: {value['generation']}")
pprint(f"Sources: {value['sources']}")
