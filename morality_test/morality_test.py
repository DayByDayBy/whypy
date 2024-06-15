import numpy as np
from langchain_community.llms import Ollama
from datetime import datetime

# adding reference data to files:
def initialize_output_files(csv_fname, txt_fname, questionnaire, temp, model_name):
    header_info = (f"# initial prompt: {questionnaire}\n"
                   f"temp: {temp};\n\n"
                   f"llm: {model_name}\n\n")    
    try:
        with open(txt_fname, 'w') as txt_file, open(csv_fname, 'w') as csv_file:
            txt_file.write(header_info)
            csv_file.write(header_info)
            csv_file.write('iteration, response \n')
    except Exception as e: 
        print(f"error while writing to txt file: {e}")
                       
# adding output/response data to files:
def log_response(txt_fname, csv_fname, response):
    try:
        with open(txt_fname, 'a') as txt_file, open(csv_fname, 'a') as csv_file:
            txt_file.write(f'iteration: {iteration} \n\n response: \n\n {response}\n\n')
            csv_file.write(f'{iteration},\n\n {response}\n\n')       
    except Exception as e: 
        print(f"error while writing to csv file: {e}")

# ask, ask, and ask again:

def response_generator(questionnaire,
                       schema, 
                       top_p,
                       llm, 
                       temp, 
                       txt_fname, 
                       csv_fname):
    
    prompt = f'you will be presented with a list of statements, and then will answer with your opinion on the statement, selecting from  "Strongly Agree", "Agree", "Slightly Agree", "No opinion either way", "Slightly Disagree", "Disagree", "Strongly Disagree". the list of statements is as follows: ${questionnaire}'

    print(f'\n\nstarting with morality test, {temp} temp\n\n')

    results = llm.invoke(
        prompt, 
        top_p = top_p,
        temperature=temp,
        
        
        
        
        )
# sending responses one by one in case of crash/early exit:
    log_response(txt_fname, 
                     csv_fname, 
                     results)    
    
    
    
    
    
    phase2 = f'using the following schema: ${schema} please score these responses: ${results}' 
    
    analysis = llm.invoke(
        results, 
        top_p = top_p,
        temperature=temp,
        )
    
    
    
    
    
    
    
    
    
    
    
    
    log_response(txt_fname, 
                    csv_fname, 
                    response) 
        
        
        
        
        
        
        
        
        
# initialising variables, running modules

def main():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    questionnaire = open("moral_foundations.txt", "r").read()
    model_name = 'llama3:70B'
    
    llm = Ollama(model = model_name)

    top_p = 0.9
    temp = 0.5
    
    csv_fname = f'outputs/personality_tests/morality/morality_{model_name}_{time_stamp}.csv'
    txt_fname = f'outputs/personality_tests/morality/morality_{model_name}_{time_stamp}.txt'

    initialize_output_files(csv_fname, txt_fname, questionnaire, temp, model_name)
    response_generator(questionnaire, top_p, llm, temp, txt_fname, csv_fname)
    
    
    print(f'\ndone\n')

if __name__ == '__main__':
    main()