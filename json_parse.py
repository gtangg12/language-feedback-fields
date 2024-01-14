import json
from llm_agent import LLMAgent
from protocols import SingleOutput

def interact_with_json(path: str, task: str):
    def frame_nerf(k) -> dict[str, SingleOutput]:
        return {label: SingleOutput(description=obj['description'], distance=obj['distance']) for label, obj in data[k][0].items()}
    agent = LLMAgent(frame_nerf)

    with open(path, 'r') as f:
        data = json.load(f)
    for k in data:
        print('analyzing frame ', k)
        print(agent.query(k, task))


if __name__ == '__main__':
    interact_with_json('outputs_reasoning.json', "find the window")