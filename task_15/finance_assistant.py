from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

loader = CSVLoader("task_15/daily_expenses.csv")
documents = loader.load()

csv_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

documents = csv_splitter.split_documents(documents)
vector_store = FAISS.from_documents(
    documents,
    GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
)

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# See full prompt at https://smith.langchain.com/hub/rlm/rag-prompt
prompt = hub.pull("rlm/rag-prompt")
qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
query = input("What do you want to know about your expenses?: ")
response = qa_chain.invoke(query)
print(response)
