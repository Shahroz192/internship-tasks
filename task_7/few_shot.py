from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


expample_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

examples = [
    {"input": "1", "output": "1"},
    {"input": "2", "output": "2"},
    {"input": "3", "output": "6"},
    {"input": "4", "output": "24"},
    {"input": "5", "output": "120"},
    {"input": "6", "output": "720"},
]

template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=expample_prompt,
    prefix="Detect Pattern and Find the value for the given input",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
    example_separator="\n",
)

template.format(input="7")
prompt = template.invoke(
    {
        "input": "10",
    }
)

print(prompt)

# response = llm.invoke(prompt)
# print(response.content)
