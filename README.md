# RAG_for_QA_local

Enhanced RAG (Retrieval Augmented Generation) for Question Answering: Dive into a hands-on implementation that leverages only local components, bypassing costly APIs. This robust example demonstrates the power of Langchain, an on-premises LLM, and a Vector Database. Highlights include crafting bespoke prompts, leveraging Sentence Transformers to tap into a plethora of Data Embedding models, and designing your unique tool tailored for seamless integration with Langchain Agents.

"What encryption does Cryptolib support?" asked on a documentation of a non-existent library:
![image](https://github.com/paryska99/RAG_for_QA_local/assets/77459670/e5632743-72d9-46d5-83ca-22b70671e706)

Required Libraries and models:
- Langchain
- ChromaDB
- Sentence Transformers
- Embedding model for Sentence Transformers (You'll find models here: https://huggingface.co/spaces/mteb/leaderboard)
- Llama-cpp-python (requires compiling manually in order to take advantage of GPUs instead of CPU-only. Further Instructions: https://github.com/abetlen/llama-cpp-python)
- Llama.cpp compatible model (You'll find models here: https://huggingface.co/TheBloke. Tested on llongorca-13b-16k)
