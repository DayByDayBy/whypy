import argparse 
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

initial_prompt = "what do you consider the most overrated virtue?"

model_name = "llama3"
iterations = 1337
frequency_penalty = np.float32(0.8)
presence_penalty = np.float32(0.9)
llm = Ollama(model = model_name)

def respo(initial_prompt, iterations):
    responses = []
    current_prompt = initial_prompt
    for i in range(iterations):
        response = llm.invoke(
            current_prompt, 
            max_tokens=100, 
            frequency_penalty = float(frequency_penalty), 
            presence_penalty = float(presence_penalty))
        responses.append([i, response, '\n\n'])
        current_prompt = response
        print(f'{iterations-i} iterations to go')
    return responses  
     
returned_responses = respo(initial_prompt, iterations)
indexed_responses = []
for ip, it in enumerate(returned_responses):
    indexed_responses.append((ip, it))

# print to file:

df_output_filename = f'outputs/prou_df_{model_name}_{time_stamp}.csv'

df = pd.DataFrame(indexed_responses, columns=['Index', 'Response'])

with open(df_output_filename, 'w') as f:
    f.write("# initial prompt:  " + initial_prompt + "\n\n") 
    df.to_csv(f, index=False)

output_filename = f'outputs/prou_{model_name}_{time_stamp}.txt'
with open(output_filename, 'w') as f:
    for index, response in indexed_responses:      
        f.write(f'iteration {index}: \n {response}\n\n')