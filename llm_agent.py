from typing import List, Literal, Optional, Tuple

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
        self.reset()
        

    def query(self, user_pose: TorchTensor[4, 4], task_prompt: str) -> str:

        # Given task prompt, LLM interprets it and determines whether or not it needs to call NERF API
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task_prompt=task_prompt) 
        prompt_response = self.gpt.forward(initial_prompt)

        prompt_json = json.loads(prompt_response)
        needs_scene_descriptions = prompt_json.needs_scene_description

        if needs_scene_descriptions:

            NERF_outputs = NERF_API(user_pose) #TODO: replace with actual NERF API call

            scene_descriptions: List[Tuple[tuple[float, float, float], str]] = [
                (value.scoord, value.description) for value in NERF_outputs.values()
            ]

            # Tasks in the list of coordinates and descriptions and interprets them for the user's task
            additional_prompt = ADDITIONAL_PROMPT_TEMPLATE.format(task_prompt=task_prompt, scene_descriptions=scene_descriptions)

            prompt_response = self.gpt.forward(additional_prompt) 
            prompt_json = json.loads(prompt_response)

            return prompt_response.output
        
        return prompt_json.output