# gpt4all
A repo which helps to finetune a gpt4all LLM based on the langcahin framework.
 clone the source code 
 create a virtual environment
 download the dependencies mentioned in the requirements.txt file
 download the LLM model from this link https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin
 run the app.py 


#workflow
1)Use langchain loaders to import the desired document.
2)Divide the documents into smaller sections or chunks
3)Convert the text into embedding which represent the semantic meaning.
4)Store the embeddings in a Db specifically Chroma DB
5)Conduct a semantic search to retrieve the most relevant based on our query.
6)Incorporate the retrieved information as context into our LLM
