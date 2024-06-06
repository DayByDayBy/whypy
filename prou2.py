import argparse 
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d_%H%M")    
prompt = "which words or phrases do you most overuse?"
model_name = "llama3"
iterations = 10
frequency_penalty = 0.8,
presence_penalty = 0.8,
llm = Ollama(model = model_name)
response = llm.invoke(
    prompt, 
    max_tokens=100, 
    temperature=0.9,
    frequency_penalty = 0.8,
    presence_penalty = 0.9)
responses = [response,]


def respo():
    i = 0
    while i < iterations:
        responsed = llm.invoke(
            response, 
            max_tokens=100, 
            frequency_penalty= 0.9, 
            presence_penalty= 0.9)
        responses.append([i, responsed, '\n\n'])
        i += 1
    return responses  
     
respo()
print(responses)




# print to file:

indexed_responses = []
for i, r in enumerate(responses):
    indexed_responses.append((i, r))

df = pd.DataFrame(indexed_responses, columns=['Index', 'Response'])
df_output_filename = f'outputs/prou2_df_{model_name}_{time_stamp}.csv'
df.to_csv(df_output_filename, index=False)

output_filename = f'outputs/prou2_{model_name}_{time_stamp}.txt'
with open(output_filename, 'w') as output_file:
    for index, response in indexed_responses:      
        output_file.write(f'iteration {index}: \n {response}\n\n')