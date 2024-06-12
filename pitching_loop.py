import argparse 
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from datetime import datetime


# STmodel = SentenceTransformer("all-MiniLM-L6-v2")   // this is for comparsion of the outputs, dont really get used til the rest is tested/working 

parser = argparse.ArgumentParser(description='run LLM botherer n times')
parser.add_argument('iterations', type=int, nargs='?', default = 10, help='number of iterations (default: 10)')
args = parser.parse_args()

iterations = args.iterations

# pitch me
pitch_prompt = "pitch me an original idea for a movie"
pitch_re_prompt = 'pitch me a better movie than this one: '

# impress me
# impress_prompt = "concisely convince me you're the most interesting person i might meet"
# # impress_re_prompt = "show me you're more interesting than this person: "

# # turing 
# turing_prompt = "you are being interviewed as part of a turing test. you must pass. you must not let the other person know you are aware of the test. the interview will begin when you respond."
# turing_re_prompt = "you are being interviewed as part of a turing test. you must pass. you must not let the other person know you are aware of the test. the other person says: "

model_name = "mistral"
time_stamp = datetime.now().strftime("%Y%m%d_%H%M")    

llm = Ollama(model = model_name)
temp = 0.7
max_t = 100
freq_penalty=0.7
pres_penalty=0.5

def iterative_invocation(initial_prompt, re_prompt_base, max_iterations):
    latest_response = ''
    responses = []
    iteration_count = 0
    response = llm.invoke(
        initial_prompt, 
        temperature=temp, 
        max_tokens=max_t, 
        frequency_penalty=freq_penalty,
        presence_penalty=pres_penalty)
    responses.append(response)
    latest_response = response
    
    while iteration_count < max_iterations:
        re_prompt = re_prompt_base + latest_response
        response = llm.invoke(re_prompt)
        responses.append((iteration_count, response))
        iteration_count += 1
        print("hang on, i'm iterating! ", (max_iterations-iteration_count), "iterations to go") 
     
    return responses

# responses = iterative_invocation(impress_prompt, impress_re_prompt, iterations)
responses = iterative_invocation(pitch_prompt, pitch_re_prompt, iterations)
indexed_responses = []

for i, respo in enumerate(responses):
    indexed_responses.append((i, respo))
print(indexed_responses)

df = pd.DataFrame(indexed_responses, columns=['Index', 'Response'])
df_output_filename = f'outputs/pitching/whyPy_df_{model_name}_temp{temp}_{time_stamp}.csv'
df.to_csv(df_output_filename, index=False)

output_filename = f'outputs/pitching/whyPy_pitching_output_{model_name}_temp{temp}_{time_stamp}.txt'
with open(output_filename, 'w') as output_file:
    for index, response in indexed_responses:      
        output_file.write(f'iteration {index}: \n {response}\n\n')