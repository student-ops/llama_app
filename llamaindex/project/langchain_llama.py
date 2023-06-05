import pickle
import os
from datetime import datetime

import openai
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from llama_index import download_loader, GPTVectorStoreIndex

download_loader("GithubRepositoryReader")
from llama_hub.github_repo import GithubClient, GithubRepositoryReader


docs = None
if os.path.exists("docs.pkl"):
    print("docs is exit")
    current_datetime = datetime.now()
    with open("pkl_update_log.txt", "a") as f:
        f.write(str(current_datetime) + "\n")
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    print("docs is None")
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner="student-ops",
        repo="raspi_go_TE",
        filter_directories=(
            ["gpt_index", "docs"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [
                ".py",
                ".go",
            ],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        verbose=True,
        concurrent_requests=10,
    )
    print(loader)

    docs = loader.load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTVectorStoreIndex.from_documents(documents=docs)
# response = query_engine.query("このレポジトリについて簡単に説明して")
# # response = index.query("Explain each LlamaIndex class?")
# print(response)
tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you need to answer questions and related github project and programming",
        return_direct=True,
    ),
]
llm = ChatOpenAI(
    temperature=0,
    client=openai,
)
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    max_token_limit=3000,
)
llm = ChatOpenAI(temperature=0)
prefix = """Anser the following questions as best you can, but speaking Japanese. You have access to the following tools:"""
suffix = """Begin! Remember to speak Japanese when giving your final answer."""

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    prefix=prefix,
    suffix=suffix,
    verbose=True,
)
string = "このgithubレポについて簡単に説明して"
agent_chain.run(input=string)
