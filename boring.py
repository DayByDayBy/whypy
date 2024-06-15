import numpy as np
from langchain_community.llms import Ollama
from datetime import datetime

# adding reference data to files:
def initialize_output_files(csv_fname, txt_fname, initial_prompt, temp, frequency_penalty, presence_penalty, model_name):
    header_info = (f"# initial prompt: {initial_prompt}\n"
                   f"temp: {temp};  FP: {frequency_penalty}; PP: {presence_penalty}\n\n"
                   f"llm: {model_name}\n\n")    
    try:
        with open(txt_fname, 'w') as txt_file, open(csv_fname, 'w') as csv_file:
            txt_file.write(header_info)
            csv_file.write(header_info)
            csv_file.write('iteration, response \n')
    except Exception as e: 
        print(f"error while writing to txt file: {e}")
                       
# adding output/response data to files:
                       
def log_response(txt_fname, csv_fname, iteration, response):
    try:
        with open(txt_fname, 'a') as txt_file, open(csv_fname, 'a') as csv_file:
            txt_file.write(f'iteration: {iteration} \n\n response: \n\n {response}\n\n')
            csv_file.write(f'{iteration},\n\n {response}\n\n')       
    except Exception as e: 
        print(f"error while writing to csv file: {e}")

# ask, ask, and ask again:

def response_generator(initial_prompt, iterations, max_t, top_p, llm, temp, frequency_penalty, presence_penalty, txt_fname, csv_fname):
    
    current_prompt = initial_prompt
        
    print(f'\n\nstarting with prompt="{initial_prompt}", {iterations} iterations, {temp} temp\n\n')

    for i in range(iterations):
        print(f'iteration {i+1}/{iterations}')
        response = llm.invoke(
            current_prompt, 
            max_tokens=max_t, 
            top_p = top_p,
            temperature=temp,
            frequency_penalty = float(frequency_penalty), 
            presence_penalty = float(presence_penalty)
            )
        log_response(txt_fname, csv_fname, i+1, response)    # sending responses one by one in case of crash/early exit
        current_prompt = response
        
        
# initialising variables, running modules

def main():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    initial_prompt = "i just heard the most boring story. can you guess what it was?"
    model_name = 'llama3:70B'
    iterations = 8
    max_t = 5000
    top_p = 0.9
    temp = 1
    frequency_penalty = np.float32(1)
    presence_penalty = np.float32(1)
    llm = Ollama(model = model_name)
    
    csv_fname = f'outputs/boring/boring_df_{model_name}_{time_stamp}.csv'
    txt_fname = f'outputs/boring/boring_{model_name}_{time_stamp}.txt'

    initialize_output_files(csv_fname, txt_fname, initial_prompt, temp, frequency_penalty, presence_penalty, model_name)
    response_generator(initial_prompt, iterations, max_t, top_p, llm, temp, frequency_penalty, presence_penalty, txt_fname, csv_fname)
    
    
    print(f'\ndone\n')

if __name__ == '__main__':
    main()