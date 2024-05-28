import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# STmodel = SentenceTransformer("all-MiniLM-L6-v2")   // this is for comparsion of the outputs, dont really get used til the rest is tested/working 

# # pitch me
# pitchPrompt = "pitch me a good, original idea for a movie"
# pitchRePrompt = 'i just heard this movie pitch:' + latestResponse + "can you do any better?"

# impress me
impress_prompt = "quickly and concisely convince me you're the most intersting person i have met"
impress_re_prompt = "show me you're more interesting than this person: "


modelName = "llama3"
currentTemp = 0.5
iteration_setting = { '1x':1, '10x':10, '100x': 100, '1000x': 1000}  # not sure about this approach, but the previous way felt clumsier, and this works fine for testing with locked values
iterations = '10x'
responses = []

# @title ask and ask

llm = Ollama(model = modelName)


def iterative_invocation(initial_prompt, re_prompt_base, max_iterations):
    latest_response = ''
    responses = []
    iteration_count = 0
    re_prompt_base = "show me you're more interesting than this person: "
    
    response = llm.invoke(initial_prompt)
    responses.append(response)
    latest_response = response
    iteration_count += 1
    
    while iteration_count < max_iterations:
        re_prompt = re_prompt_base + latest_response
        response = llm.invoke(re_prompt)
        responses.append(response)
        iteration_count += 1
    
    return responses


responses = iterative_invocation(impress_prompt, impress_re_prompt, iterations)

indexed_responses = []

for i, respo in enumerate(responses):

    indexed_responses.append((i, respo))


# print(llm.invoke(impressPrompt))
# print(llm.invoke(pitchPrompt))



