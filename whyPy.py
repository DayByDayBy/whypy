import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

STmodel = SentenceTransformer("all-MiniLM-L6-v2")

# pitch me
pitchPrompt = "pitch me a good, original idea for a movie"
latestResponse = ''
newPitchPrompt = 'i just heard this movie pitch:' + latestResponse + "can you do any better?"

# impress me
impressPrompt = "quickly and concisely convince me you are the most intersting person i have met"
latestResponse = ''
newImpressPrompt = 'i just met this person:' + latestResponse + "prove to me you are more interesting"


modelName = "llama3"
currentTemp = 0.5

iterationSetting = { '1x':1, '10x':10, '100x': 100, '1000x': 1000}  # not sure about this approach, but the previous way felt clumsier, and this works fine for testing with locked values
iterationCount = 0
responses = []

# @title ask and ask

llm = Ollama(model = modelName)

print(llm.invoke(impressPrompt))
print(llm.invoke(pitchPrompt))



