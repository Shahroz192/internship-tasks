from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

template = ChatPromptTemplate(
    [
        ("system", "You are an assistant designed to help users with the specifications of smartphones form the company based on the {content}."),
        ("human", "Tell me about the {text}"),
    ]
)

text = input("Enter the smartphone model: ")

content = "samsung galaxy s21 ultra is a smartphone model with a 6.8-inch display, 108MP camera, 5000mAh battery, and 13GB RAM and 100 GB storage." \
"iphone 13 pro max is a smartphone model with a 6.7-inch display, 12MP camera, 4352mAh battery, and 9GB RAM and 128 GB storage." \
"google pixel 6 pro is a smartphone model with a 6.7-inch display, 50MP camera, 9000mAh battery, and 12GB RAM and 128 GB storage."
"oneplus 9 pro is a smartphone model with a 6.7-inch display, 48MP camera, 4500mAh battery, and 12GB RAM and 1000 GB storage."
prompt = template.invoke(
    {
        "text": text,
        "content": content,
    }
)
response = model.invoke(prompt)
print(response.content)
