from typing import List, Literal, Optional, Tuple

import torch
from torchtyping import TensorType as TorchTensor

from language_feedback_field.model_gpt import SystemMode, GPT
from prompts import SYSTEM_PROMPT_TEMPLATE, INITIAL_PROMPT_TEMPLATE, ADDITIONAL_PROMPT_TEMPLATE

import json

class LLMAgent():

    def __init__(
        self
    ):
        """
        :param task_prompt: prompt describing agent's task 
        :param system_mode: system config mode
        """
        super().__init__()
        self.gpt = GPT(system_text=SYSTEM_PROMPT_TEMPLATE, system_mode=SystemMode.JSON)
        

    def query(self, user_pose: TorchTensor[4, 4], task_prompt: str) -> str:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Given task prompt, LLM interprets it and determines whether or not it needs to call NERF API
                initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task_prompt=task_prompt) 
                prompt_response = self.gpt.forward(text=initial_prompt, temperature=max_attempts * 0.1)
                prompt_json = json.loads(prompt_response)
                needs_scene_descriptions = prompt_json['needs_scene_descriptions']
                break 
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt == max_attempts - 1:
                    raise


        if needs_scene_descriptions:
            NERF_outputs = NERF_API(user_pose) #TODO: replace with actual NERF API call

            scene_descriptions: List[Tuple[tuple[float, float, float], str]] = [
                (value.scoord, value.description) for value in NERF_outputs.values()
            ]
            # Test value: [((2.874, 2.108, 4.092), "Dark brown table"), ((3.062, 3.138, 2.894), "White fridge"), ((3.062, 3.138, 2.894), "Red carpet")]

            for attempt in range(max_attempts):
                try:
                    # Takes in the list of coordinates and descriptions and interprets them for the user's task
                    additional_prompt = ADDITIONAL_PROMPT_TEMPLATE.format(task_prompt=task_prompt, scene_descriptions=scene_descriptions)
                    prompt_response = self.gpt.forward(text=additional_prompt, temperature=max_attempts * 0.1) 
                    prompt_json = json.loads(prompt_response)
                    return prompt_json['output']
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt == max_attempts - 1:
                        raise
        
        return prompt_json['output']
    

    
# Comment out or remove later!
if __name__ == '__main__':
    agent = LLMAgent()
    tensor = torch.Tensor(4, 4)
    tensor.fill_(1)

    agent_output = agent.query(user_pose=tensor, task_prompt="I want to walk on something fuzzy.")
    print(agent_output)

    agent_output = agent.query(user_pose=tensor, task_prompt="I'm thirsty.")
    print(agent_output)