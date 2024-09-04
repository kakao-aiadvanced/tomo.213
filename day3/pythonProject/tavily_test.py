from tavily import TavilyClient
tavily = TavilyClient(api_key='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')

response = tavily.search(query="Where does Messi play right now?", max_results=3)
print(response)
context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]

# You can easily get search result context based on any max tokens straight into your RAG.
# The response is a string of the context within the max_token limit.

response_context = tavily.get_search_context(query="Where does Messi play right now?", search_depth="advanced", max_tokens=500)
print(response_context)

# You can also get a simple answer to a question including relevant sources all with a simple function call:
# You can use it for baseline
response_qna = tavily.qna_search(query="Where does Messi play right now?")
print(response_qna)