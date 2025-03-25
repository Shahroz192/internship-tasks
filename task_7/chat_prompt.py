from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Tell me about the topic {text}"),
    ]
)


text = input("Enter the topic: ")
prompt = template.invoke(
    {
        "text": text,
    }
)
response = llm.invoke(prompt)
print(response.content)
