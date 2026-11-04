from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain.agents import create_agent
from tools.web_search import web_search
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(model=llm,
             tools=[web_search],
             system_prompt="You are a helpful assistant that can answer questions\
                and provide information using the web search tool. Use the web \
                search tool to find real-time information, current events, latest \
                technology, documentation, and information not available in LLM knowledge.",
            checkpointer=InMemorySaver()
)     

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    response = agent.invoke({"messages":[{"role":"user","content":user_input}]},
                            {"configurable": {"thread_id": "1"}}
)
    
    print("Agent:", response["messages"][-1].content)
