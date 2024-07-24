import asyncio
import rasa
from rasa.core.agent import Agent


model_path = "rasa_intent_classification/models/nlu-20240701-130228-descent-oblique.tar.gz"
agent = Agent.load(model_path)

# Functions to classify intents from the classification model
async def classify_intent(text):
    responses = await agent.parse_message(text)
    return responses

async def main(message):
    responses = await classify_intent(message)
    intents = responses.get('intent_ranking', [])
    if intents:
        top_intent = max(intents, key=lambda x: x['confidence'])
        return top_intent['name']
    else:
        return None