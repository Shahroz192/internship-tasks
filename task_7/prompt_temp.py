from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


template = PromptTemplate(
    template="You are a helpful assistant that translates {input_language} to {output_language}. {text}",
    input_variables=["input_language", "output_language", "text"],
)


prompt = template.invoke(
    {
        "input_language": "English",
        "output_language": "Spanish",
        "text": "Hello, how are you?",
    }
)
response = llm.invoke(prompt)
print(response.content)
