from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
from langchain.llms import CTransformers

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.01})



# llama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="codellama")
df = SmartDataframe("data/customerdata.csv", config={"llm": llm})

print(df.chat('How many customers are there with Female gender?'))