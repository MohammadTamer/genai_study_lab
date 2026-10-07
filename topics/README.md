# GenAI Study Lab - Topics

Standalone, modular Python scripts to understand each GenAI and LangChain concept independently.
Every file has its own `main()` function, self-contained configuration, and runs 100% locally with no paid API keys.

---

## Files and Topics

| # | File | Topic Description | Run Command |
| :---: | :--- | :--- | :--- |
| **01** | [`01_basic_prompt.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/01_basic_prompt.py) | **Basic & Zero-Shot**: ChatPromptTemplate, System/Human roles, LCEL. | `python topics/01_basic_prompt.py` |
| **02** | [`02_few_shot.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/02_few_shot.py) | **Few-Shot Learning**: FewShotChatMessagePromptTemplate for sentiment classification. | `python topics/02_few_shot.py` |
| **03** | [`03_chains_and_lcel.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/03_chains_and_lcel.py) | **LCEL & Sequential Chains**: Pipe `\|` operator, `RunnableLambda`, sequential execution. | `python topics/03_chains_and_lcel.py` |
| **04** | [`04_parallel_chain.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/04_parallel_chain.py) | **RunnableParallel**: Executing multiple branches (summary + key points) simultaneously. | `python topics/04_parallel_chain.py` |
| **05** | [`05_rag.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/05_rag.py) | **Modern LCEL RAG**: Chunking, InMemoryVectorStore, grounded prompt with LCEL. | `python topics/05_rag.py` |
| **05b** | [`05b_rag_classic_chains.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/05b_rag_classic_chains.py) | **Classic Chains RAG**: RetrievalQA and ConversationalRetrievalChain with memory. | `python topics/05b_rag_classic_chains.py` |
| **06** | [`06_self_consistency.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/06_self_consistency.py) | **Self-Consistency**: Generating multiple reasoning paths and majority voting. | `python topics/06_self_consistency.py` |
| **07** | [`07_tools.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/07_tools.py) | **Custom Tools**: Creating safe tools using `@tool` decorator and AST parser. | `python topics/07_tools.py` |
| **08** | [`08_single_agent.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/08_single_agent.py) | **Single Agent**: Autonomous agent choosing and executing tools. | `python topics/08_single_agent.py` |
| **09** | [`09_multi_agent.py`](file:///c:/Users/User/Desktop/genai_study_lab/topics/09_multi_agent.py) | **Multi-Agent System**: Manager agent delegating to research and math specialists. | `python topics/09_multi_agent.py` |

---

## Local Setup

Make sure Ollama is installed and running:
```bash
ollama serve
ollama pull llama3.1
ollama pull nomic-embed-text
```
