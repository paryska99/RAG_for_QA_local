# === IMPORTS ===
# External libraries
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

# Internal resources
import my_documentation

# === USER INPUT ===
user_input = input()

# === TEMPLATE DEFINITION ===
# This is the template used by the LLM to produce a response.
template_with_history = """
Answer the following questions using the documentation available to you. You have access to the following tools and documents:
{tools}

Use the following format and only the following format, stick to it the best you can and fill out every step:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, there is currently one tool, it should be [{tool_names}]
Action Input: the input to the action
Observation: the result of the action, likely a snippet from the documentation
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer based on the documentation
Final Answer: the final answer to the original input question, backed by the documentation

Example:
Question: what is Cryptolib? Explain please
Thought: I should search for this information.
Action: Search Documentation
Action Input: What is Cryptolib?
Observation: Cryptolib is ...
Thought: I now know...
Final Answer: Cryptolib is ...

Begin!
New question: {input}
{agent_scratchpad}
"""

# === CUSTOM TOOL CREATION ===
# This tool is a wrapper around your documentation search function.
custom_tool = Tool(
    name="Search Documentation",
    func=my_documentation.get_relevant_documents,
    description="Useful for when you need to answer questions using our own database with documentation about our software. After you pick the tool, write the question you'd like to ask the database in the action input. Example: What is Cryptolib? Example: What does x function do?"
)
tool_names = "Search Documentation"

# === LLM SETUP ===
# This block sets up the Llama Language Model.
llamallm = LlamaCpp(
    model_path="C:\\Language_Model_Alpaca\\LLAMA14-05-2023\\llama.cpp\\models\\llongorca-13b-16k.Q3_K_M.gguf",
    n_gpu_layers=14,
    n_batch=512,  # Adjust based on available RAM
    n_ctx=4096,  # Set max token count
    verbose=True,
    streaming=True,
    temperature=0.4
)

# Create a prompt template for the LLM
prompt = PromptTemplate(template=template_with_history, input_variables=["input", "tools", "tool_names", "agent_scratchpad"])

# === AGENT SETUP ===
# This block sets up the ZeroShotAgent and its executor.
llm_chain = LLMChain(llm=llamallm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=[custom_tool], verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=[custom_tool], verbose=True
)

# === RUN THE AGENT CHAIN ===
agent_chain.run(input=user_input, tool_names=tool_names, tools=custom_tool)
