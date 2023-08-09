# Import necessary libraries
import pandas as pd
from datasets import load_dataset
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings

# # Download the dataset
# dataset = load_dataset('medical_dialog', 'processed.en')
# df = pd.DataFrame(dataset['train'])
# print(df.head())

# # Save the data to a file
# dialog = []
# patient, doctor = zip(*df['utterances'])
# for i in range(len(patient)):
#     dialog.append(patient[i])
#     dialog.append(doctor[i])
# dialog_df = pd.DataFrame({"dialog": dialog})
# dialog_df.to_csv('data.txt', sep=" ", index=False)

# Document embedding and store into chroma DB 
loader = TextLoader('data.txt')
index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings()).from_loaders([loader])

# Initialize GPT4All
local_path = './ggml-gpt4all-j-v1.3-groovy.bin'  # Replace with your desired local file path
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, backend='gptj')

# Similarity search
query = "what is the solution for sore throat"
results = index.vectorstore.similarity_search(query, k=4)
context = "\n".join([document.page_content for document in results])
print(f"Retrieving information related to your question...")
print(f"Found this content which is most similar to your question:\n{context}")

# Using langchaintemplate
template = """ 
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial (context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)
# Print the result
print("Processing the information with gpt4all...\n")
print(llm_chain.run("what is the solution for sore throat"))

