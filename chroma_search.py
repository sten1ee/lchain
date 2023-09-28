import os
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


# load the document and split it into chunks
loader = TextLoader("bodhisattva_vow.txt")
documents = loader.load()


# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

PATH_TO_DB = "./chroma_db"
if not os.path.exists(PATH_TO_DB):
    # split it into chunks
    # and save to disk upon creation:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding_function, persist_directory=PATH_TO_DB)
else:
    # load from disk a prebuild index:
    db = Chroma(persist_directory=PATH_TO_DB, embedding_function=embedding_function)

qnumber = 0
while True:
    qnumber += 1
    # query it & print results
    query = input(f"Q[{qnumber}] >> ")
    if not query:
        break
    hits = db.similarity_search(query)
    print(f"A[{qnumber}] << {hits[0].page_content}")

