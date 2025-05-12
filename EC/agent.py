import carla
import time
import torch
import math
import os
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18 + 1, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 3),  # steer, throttle, brake
            nn.Tanh()  # steer ∈ [-1,1], throttle & brake 可 clamp
        )

    def forward(self, image, velocity):
        x = self.features(image)  # [B, C, H, W]
        x = x.view(x.size(0), -1)  # → [B, N]
        velocity = velocity.view(velocity.size(0), -1)  # → [B, 1]
        x = torch.cat((x, velocity), dim=1)
        out = self.classifier(x)
        steer, throttle, brake = out[:, 0], out[:, 1], out[:, 2]
        return steer, throttle.clamp(0, 1), brake.clamp(0, 1)

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PilotNet().to(self.device)
        self.model.load_state_dict(torch.load("px141_scen_v2.pth", map_location=self.device))
        self.model.eval()
        

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img[190:331, :, :]),
            transforms.ToPILImage(),
            transforms.Resize((66, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])


        self.frame_count = 0
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)



    def visualize_saliency(model, image_tensor, velocity_tensor):
        # image_tensor.requires_grad = True is required
        image_tensor.requires_grad_()

        # Forward pass
        steer, throttle, brake = model(image_tensor, velocity_tensor)
        output = steer.sum()
        output.backward()

        # Get gradients (with respect to the input)
        saliency = image_tensor.grad.data.abs()
        saliency, _ = torch.max(saliency, dim=1)          # Take max across channels, shape becomes [batch_size, H, W]
        saliency_map = saliency.squeeze().cpu().numpy()
        return saliency_map



    def run_step(self, img, vel):
        STEER_FULL   = 0.10          # Threshold for straight road
        STEER_CUTOFF = 0.35          # Minimum speed threshold
        V_MAX = 17                # m/s
        V_MIN = 4.0                  # m/s


        def speed_from_steer(s):
            a = abs(s)
            if a <= STEER_FULL:
                return V_MAX
            elif a >= STEER_CUTOFF:
                return V_MIN
            else:
                ratio = (a - STEER_FULL) / (STEER_CUTOFF - STEER_FULL)
                return V_MAX - ratio * (V_MAX - V_MIN)

        def throttle_from_steer(s):
            a = abs(s)
            if a <= STEER_FULL:
                return 1.0
            elif a >= STEER_CUTOFF:
                return 0.30
            else:
                ratio = (a - STEER_FULL) / (STEER_CUTOFF - STEER_FULL)
                return 1.0 - 0.60 * ratio 
            

        # —— 1. Predict steering angle ——
        image = self.transform(img).unsqueeze(0).to(self.device)
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        velocity = torch.tensor([[speed]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            steer, throttle_BETA, brake_BETA = self.model(image, velocity)

        steer_val = float(steer.clamp(-1, 1))

        # —— 2. Steering → Target speed & Base throttle ——
        target_speed   = speed_from_steer(steer_val)
        base_throttle  = throttle_from_steer(steer_val)

        # Prevent simultaneous throttle and brake
        err = target_speed - speed           # m/s
        if err > 0:
            throttle = min(1.0, base_throttle + 0.05 * err)
            brake    = 0.0
        else:
            throttle = base_throttle * 0.5                   
            brake    = min(1.0, 0.1 * (-err))                

        if throttle > 0 and brake > 0:
            brake = 0.0 if brake < 0.3 else brake
            throttle = 0.0 if brake >= 0.3 else throttle

        # —— 4. Output control ——
        control = carla.VehicleControl()
        control.steer    = steer_val
        control.throttle = throttle
        control.brake    = brake
        return control
