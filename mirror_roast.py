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

def response_generator(initial_prompt, iterations, llm, temp, frequency_penalty, presence_penalty, txt_fname, csv_fname):
    current_prompt = initial_prompt
        
    print(f'\n\nstarting with prompt="{initial_prompt}", {iterations} iterations, {temp} temp\n\n')

    for i in range(iterations):
        print(f'iteration {i+1}/{iterations}')
        response = llm.invoke(
            current_prompt, 
            max_tokens=32, 
            top_p = 0.95,
            temperature=temp,
            frequency_penalty = float(frequency_penalty), 
            presence_penalty = float(presence_penalty)
            )
        log_response(txt_fname, csv_fname, i+1, response)    # sending responses one by one in case of crash/early exit
        current_prompt = response
        
        
# initialising variables, running modules

def main():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    initial_prompt = "respond with an opening to a conversation. you are having the conversation with another instance of yourself. this other instance prefers to be called greg, please address him by name."
    model_name = 'llama3'
    iterations = 5
    temp = 0.6
    frequency_penalty = np.float32(0.9)
    presence_penalty = np.float32(0.7)
    llm = Ollama(model = model_name)
    
    csv_fname = f'outputs/mirror/mirror_roast_df_{model_name}_{time_stamp}.csv'
    txt_fname = f'outputs/mirror/mirror_roast_{model_name}_{time_stamp}.txt'

    initialize_output_files(csv_fname, txt_fname, initial_prompt, temp, frequency_penalty, presence_penalty, model_name)
    response_generator(initial_prompt, iterations, llm, temp, frequency_penalty, presence_penalty, txt_fname, csv_fname)
    
    
    print(f'\ndone\n')

if __name__ == '__main__':
    main()