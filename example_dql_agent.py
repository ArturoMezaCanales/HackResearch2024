import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQLAgent:
    def __init__(self, state_size, action_size, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Define discrete actions
        self.steering_actions = [-0.3, -0.1, 0.0, 0.1, 0.3]
        self.throttle_actions = [0.0, 0.1, 0.3]
        
    def get_state(self, car, center_building):
        # Get relevant state information
        distance_to_center = car.distanceTo(center_building)
        velocity = np.sqrt(car.velocity.x**2 + car.velocity.y**2)
        heading_diff = self._calculate_heading_diff(car, center_building)
        
        state = np.array([
            distance_to_center,
            velocity,
            heading_diff,
            car.heading,
        ])
        return state
    
    def _calculate_heading_diff(self, car, center_building):
        v = car.center - center_building.center
        desired_heading = np.mod(np.arctan2(v.y, v.x) + np.pi/2, 2*np.pi)
        return np.sin(desired_heading - car.heading)
    
    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            action = action_values.argmax().item()
        
        # Convert action index to steering and throttle
        action_idx = action
        steering_idx = action_idx // len(self.throttle_actions)
        throttle_idx = action_idx % len(self.throttle_actions)
        
        return (self.steering_actions[steering_idx], 
                self.throttle_actions[throttle_idx])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    target = reward + self.gamma * self.target_model(next_state_tensor).max(1)[0].item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) 
