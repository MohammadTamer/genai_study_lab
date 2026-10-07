# Course Notes: Generative AI

## LLM
A Large Language Model is a language-focused model trained on large amounts of text. It can understand and generate human-like language.

## Prompt Engineering
Prompt engineering designs instructions and context to guide model outputs. Clear role, task, context, constraints, and output format usually improve reliability.

## Zero-Shot
Zero-shot prompting asks the model to perform a task without examples.

## Few-Shot
Few-shot prompting gives the model a small number of examples inside the prompt so it can infer the expected pattern.

## In-Context Learning
In-context learning uses the examples and instructions present in the current context without changing the model's parameters.

## RAG
Retrieval-Augmented Generation retrieves relevant external information and supplies it to the model before generation. This can ground answers in current or private data.

## Retriever
A retriever finds the most relevant documents or chunks for a query.

## Vector Database
A vector store keeps embeddings and supports semantic similarity search. Similar meanings can be retrieved even when the wording differs.

## Agent
An agent combines a model with tools so it can decide which tools to use and iteratively work toward a goal.

## Multi-Agent System
A multi-agent system uses multiple agents with different roles that cooperate on a larger task.

## LCEL
LangChain Expression Language uses the pipe operator to connect runnable components. It makes data flow explicit and composable.

## RunnableLambda
RunnableLambda wraps a normal Python function so it can participate in an LCEL chain.

## RunnableParallel
RunnableParallel runs multiple runnable branches using the same input and returns a keyed result.

## Hallucination Mitigation
Grounding with retrieved evidence, constrained prompts, structured outputs, and evaluation are common ways to reduce unsupported model claims.

## Fine-Tuning
Fine-tuning updates a pre-trained model using task- or domain-specific training examples. It changes model parameters, unlike prompt-only in-context learning.
