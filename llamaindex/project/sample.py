import openai
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader


documents = SimpleDirectoryReader("./data/").load_data()
index = GPTVectorStoreIndex.from_documents(documents=documents)

tools = [
    Tool(
        name="LlamaIndex",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you need to answer questions about graham",
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
    max_token_limit=1000,
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
agent_chain.run(input="What did graham do growing up?")
