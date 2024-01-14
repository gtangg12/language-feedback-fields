from babies import mock_nerf_babies, babies_task, front_pose, back_pose
from llm_agent import LLMAgent
from protocols import LLMModule
import pygame
from spatial_cache import SpatialCache
import torch
from torchtyping import TensorType as TorchTensor

print('imports successful')
class App():
    def __init__(self, llm: LLMModule, task: str):
        self.window_width = 800
        self.window_height = 600
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        self.pose: TorchTensor[4, 4] = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32) # world2cam, need to invert it to get cam2world
        self.llm = llm
        self.task = task
        
        pygame.init()

    def run(self):
        # preprocess the images and load them 
        self.render()
        pygame.display.update()
        while True:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    pygame.quit()
                    break
            key_pressed = pygame.key.get_pressed()
            translate = 0.1
            rotate = torch.tensor([0.02])
            if key_pressed[pygame.K_w]:
                # forward
                self.pose[2, 3] += translate
            if key_pressed[pygame.K_s]:
                # backward
                self.pose[2, 3] -= translate
            if key_pressed[pygame.K_a]:
                # left
                self.pose[0, 3] -= translate
            if key_pressed[pygame.K_d]:
                # right
                self.pose[0, 3] += translate
            if key_pressed[pygame.K_q]:
                # up
                self.pose[1, 3] += translate
            if key_pressed[pygame.K_e]:
                # down
                self.pose[1, 3] -= translate
            if key_pressed[pygame.K_i]:
                # pitch up
                new_r = self.pose[:, :3] @ torch.tensor([
                    [torch.cos(rotate), 0, torch.sin(rotate)],
                    [0, 1, 0],
                    [-torch.sin(rotate), 0, torch.cos(rotate)],
                ])
                self.pose[:, :3] = new_r
            if key_pressed[pygame.K_k]:
                # pitch down
                new_r = self.pose[:, :3] @ torch.tensor([
                    [torch.cos(-rotate), 0, torch.sin(-rotate)],
                    [0, 1, 0],
                    [-torch.sin(-rotate), 0, torch.cos(-rotate)],
                ])
                self.pose[:, :3] = new_r
            if key_pressed[pygame.K_j]:
                # yaw left
                new_r = self.pose[:, :3] @ torch.tensor([
                    [1, 0, 0],
                    [0, torch.cos(rotate), -torch.sin(rotate)],
                    [0, torch.sin(rotate), torch.cos(rotate)],
                ])
                self.pose[:, :3] = new_r
            if key_pressed[pygame.K_l]:
                # yaw right
                new_r = self.pose[:, :3] @ torch.tensor([
                    [1, 0, 0],
                    [0, torch.cos(-rotate), -torch.sin(-rotate)],
                    [0, torch.sin(-rotate), torch.cos(-rotate)],
                ])
                self.pose[:, :3] = new_r
            if key_pressed[pygame.K_u]:
                # roll left
                new_r = self.pose[:, :3] @ torch.tensor([
                    [torch.cos(rotate), -torch.sin(rotate), 0],
                    [torch.sin(rotate), torch.cos(rotate), 0],
                    [0, 0, 1],
                ])
                self.pose[:, :3] = new_r
            if key_pressed[pygame.K_o]:
                # roll right
                new_r = self.pose[:, :3] @ torch.tensor([
                    [torch.cos(-rotate), -torch.sin(-rotate), 0],
                    [torch.sin(-rotate), torch.cos(-rotate), 0],
                    [0, 0, 1],
                ])
                self.pose[:, :3] = new_r
                
            self.render()
            pygame.display.flip()
            pygame.time.delay(100)

    def render(self):
        font_size = 20
        font = pygame.font.Font(None, font_size) 
        llm_output = self.llm.query(user_pose=self.pose, task_prompt=self.task)
        image = font.render(str(llm_output), True, (255, 255, 255))
        image_surface = pygame.Surface((self.window_width, self.window_height))
        image_surface.blit(image, (0, 0)) 
        self.window.blit(image_surface, (0, 0))

        print(self.pose, llm_output)

if __name__ == "__main__":
    agent_babies = LLMAgent(mock_nerf_babies)
    cached_agent = SpatialCache(agent_babies)
    app = App(cached_agent, babies_task)
    app.run()