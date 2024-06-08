import time
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
# from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

initial_prompt = "you'd better explain yourself"

model_name = "llama3"
iterations = 108
temp = 0.6
frequency_penalty = np.float32(0.9)
presence_penalty = np.float32(0.9)
llm = Ollama(model = model_name)
df_output_filename = f'outputs/explain/explain_df_{model_name}_{time_stamp}.csv'
output_filename = f'outputs/explain/explain_{model_name}_{time_stamp}.txt'

with open(output_filename, 'w') as txt:
    txt.write(f"# initial prompt: {initial_prompt}\n\n"
              f"temp: {temp};  FP: {frequency_penalty}; PP: {presence_penalty}\n\n")                        
             
with open(df_output_filename, 'w') as csv:
    csv.write(f"# initial prompt: {initial_prompt}\n\n"
              f"temp: {temp};  FP: {frequency_penalty}; PP: {presence_penalty}\n\n") 


def respo(initial_prompt, iterations):
    responses = []
    current_prompt = initial_prompt
    
    print(f'starting with prompt="{initial_prompt}", {iterations} iterations, {temp} temp\n')

    for i in range(iterations):
        print(f'iteration {i+1}/{iterations}')
        response = llm.invoke(
            current_prompt, 
            max_tokens=98, 
            temperature=temp,
            frequency_penalty = float(frequency_penalty), 
            presence_penalty = float(presence_penalty)
            )
        responses.append([i, response, '\n\n'])
        current_prompt = response
        
        indexed_responses = [(ip, it) for ip, it in enumerate(responses)]
        df = pd.DataFrame(indexed_responses, columns=['Index', 'Response'])
        
        try:  
            df.to_csv(df_output_filename, index=False, mode='a')
        except Exception as e:
            print(f"error while writing to csv file: {e}")
            
    return responses, indexed_responses

returned_responses, indexed_responses = respo(initial_prompt, iterations)
    
try: 
    with open(output_filename, 'a') as f:
        for index, response in indexed_responses:      
            f.write(f'iteration {index}: \n {response}\n\n')
except Exception as e: 
    print(f"error while writing to txt file: {e}")

print('done')