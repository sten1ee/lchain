from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#from langchain.document_loaders import CSVLoader
#import pandas as pd
#import os

#loader0 = WebBaseLoader("https://www.cs.utexas.edu/users/EWD/transcriptions/EWD10xx/EWD1036.html") # On the cruelty of really teaching computing science
#loader1 = WebBaseLoader("https://gist.githubusercontent.com/sten1ee/c9047416d132bec4f793f99ebe5c3511/raw/") # The Bodhisattva Vow

loaders = [
    WebBaseLoader("https://unfetteredmind.org/ganges-mahamudra-class/"),
    WebBaseLoader("https://unfetteredmind.org/ganges-mahamudra-class-2/"),
    WebBaseLoader("https://unfetteredmind.org/ganges-mahamudra-class-3/"),
    WebBaseLoader("https://unfetteredmind.org/ganges-mahamudra-class-4/"),
    WebBaseLoader("https://unfetteredmind.org/ganges-mahamudra-class-5/"),
]


# Create an index using the loaded documents
docsearch = VectorstoreIndexCreator().from_loaders(loaders)

# Create a question-answering chain using the index
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
