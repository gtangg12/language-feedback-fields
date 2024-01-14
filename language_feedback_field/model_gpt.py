import os
from enum import Enum
from typing import List, Literal, Optional

import requests
import torch
import torch.nn as nn
from PIL import Image

# from conceptfields.utils.formats import encode_base64
import io
import base64
from PIL import Image


def encode_base64(image: Image.Image):
    """
    Save PIL image to base64 string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')



class SystemMode(Enum):
    MAIN = 'main' # text and image capable
    JSON = 'json' # text capable supporting json
    CODE = 'code'
    USER = 'user' # custom user input


OPENAI_DEFAULT_MODEL = {
    SystemMode.MAIN: 'gpt-4-vision-preview',
    SystemMode.JSON: 'gpt-4-1106-preview', # vision-preview. gpt-4 doesn't support json yet
}

OPENAI_DEFAULT_SYSTEM_TEXTS = {
    SystemMode.MAIN: 'You are a helpful assistant.',
    SystemMode.JSON: 'You are a helpful assistant designed to output JSON.',
}


class GPT(nn.Module):
    """
    GPT endpoint with conversation history.
    """
    def __init__(
        self,
        model      : Optional[str]=None,
        system_text: Optional[str]=None,
        system_mode: SystemMode=SystemMode.MAIN,
    ):
        """
        :param model: name of OpenAI model to use
        :param system_text: text to seed the conversation: GPT has SYSTEM (seed), ASSISTANT (llm), and USER (prompt)
        :param system_mode: system config mode
        """
        super().__init__()
        assert system_mode != SystemMode.USER or (model is not None and system_text is not None), \
            'Must provide model and system text for custom system mode.'
        self.model       = model       or OPENAI_DEFAULT_MODEL       [system_mode]
        self.system_text = system_text or OPENAI_DEFAULT_SYSTEM_TEXTS[system_mode]
        self.system_mode = system_mode
        self.reset()

    def forward(self, text: Optional[str]=None, image: Optional[List[Image.Image]]=None, max_tokens=512, temperature: Optional[float]=0.0) -> str:
        """
        Send a prompt to the model and return the response.

        :param prompt: prompt to send to the model
        :param image: optional image(s) to send to the model
        :param return_response: whether to return the full response object
        """
        assert text or image
        if image is not None and not isinstance(image, list):
            image = [image]
        if image is None:
            self.process_text(text)
        else:
            self.process_image(text, image)

        headers = {
            'Content-Type' : 'application/json',
            'Authorization': 'Bearer {}'.format(os.getenv('OPENAI_API_KEY')),
        }
        data = {'model': self.model, 'messages': self.messages, 'max_tokens': max_tokens, 'temperature': temperature}
        if self.system_mode == SystemMode.JSON:
            data['response_format'] = {'type': 'json_object'}
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response = response.json()
        try:
            response_message = response['choices'][0]['message']['content']
        except:
            print(response)
            response_message = 'Sorry, I do not understand.'
        self.messages.append({'role': 'assistant', 'content': response_message})
        return response_message
    
    def process_text(self, text: str, json=False):
        """
        Add text to the next message.
        """
        self.messages.append({'role': 'user', 'content': text})

    def process_image(self, text: Optional[str], image: List[Image.Image]):
        """
        Add text and corresponding image(s) to the next message.
        """
        content = []
        if text is not None:
            content.append({'type': 'text', 'text': text})
        for x in map(encode_base64, image):
            content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{x}'}})
        self.messages.append({'role': 'user', 'content': content})
    
    def reset(self):
        """
        Clear the conversation history.
        """
        self.messages = [{'role': 'system', 'content': self.system_text}]


if __name__ == '__main__':
    llm = GPT(system_mode=SystemMode.MAIN)
    message = llm("Hi, how are you? I have a rubber blue ball.")
    print(message)
    print(llm.messages)
    message = llm("I am doing well. What color is my ball and what is it made of based on what I just said.")
    print(message)
    print(llm.messages)
    llm.reset()
    print(llm.messages)

    image = Image.open('/data/vision/torralba/scratch/gtangg12/conceptfields/tests/example1.png')
    image = image.convert('RGB')
    print(image)
    message = llm("Caption this image.", image=image)
    print(message)
    message = llm(image=image)
    print(message)