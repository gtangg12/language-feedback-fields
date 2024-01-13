from typing import List, Literal, Optional

from language_feedback_field.model_gpt import SystemMode, GPT
from prompts import SINGLE_TURN_MODE_SYSTEM_PROMPT

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

        self.gpt = GPT(system_text=SINGLE_TURN_MODE_SYSTEM_PROMPT, system_mode=SystemMode.JSON)
        self.reset()
        

    def query(self, user_pos, task_prompt) -> str:

        # Given task prompt, LLM interprets it and determines whether or not it needs to call NERF API
        #TODO: prompt engineer a good prompt template to place the task_prompt
        initial_prompt = INITIAL_PROMPT.format(task_prompt=task_prompt) 
        prompt_response = self.gpt.forward(initial_prompt)

        #TODO: replace with json extraction of prompt_response # read the json to get this boolean / ensure the LLM returns a json in this format
        prompt_json = json.loads(prompt_response)
        needs_scene_descriptions = prompt_json.needs_scene_description

        if needs_scene_descriptions:
            scene_descriptions = NERF_API(user_pos) #TODO: replace with actual NERF API call

            # TODO: prompt engineer something to aggregate the list of descriptions (scene_description)
            additional_prompt = ADDITIONAL_PROMPT.format(task_prompt=task_prompt, scene_description=scene_descriptions)

            prompt_response = self.gpt.forward(additional_prompt) 
            prompt_json = json.loads(prompt_response)

            # TODO: ensure JSON output stores the final response correctly
            return prompt_response.output
        
        return prompt_json.output

"""
prompt_template = 
        You are an agent moving through a 3D space with the the following task: {task_prompt}.

        You are given the following list of descriptions and distances/orientations of objects in your field of vision.
        {objects_list}.

        Based on the task, aggregate the descriptions and orientations you are given to produce an optimal response and direction/angle to move in. 

        Return the optimal response and direction/angle in json format.
        """