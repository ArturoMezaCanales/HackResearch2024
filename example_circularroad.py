import numpy as np
import torch
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time
from tkinter import *
from example_dql_agent import DQLAgent
import argparse
import os
import glob

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

w = World(dt, width = world_width, height = world_height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

def update_car_position(car, center_x, center_y, radius, speed, dt):
    # Get current angle
    current_angle = np.arctan2(car.center.y - center_y, car.center.x - center_x)
    
    # Update angle based on speed and radius
    angular_velocity = speed / radius  # v = ω * r
    new_angle = current_angle + angular_velocity * dt
    
    # Calculate new position
    new_x = center_x + radius * np.cos(new_angle)
    new_y = center_y + radius * np.sin(new_angle)
    
    # Update car position and heading
    car.center = Point(new_x, new_y)
    car.heading = new_angle + np.pi/2  # Add π/2 to make car tangent to circle

def environment_setup(w):
    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
    # A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

    # To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
    cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
    w.add(cb)
    rb = RingBuilding(Point(world_width/2, world_height/2), inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
    w.add(rb)

    # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
    for lane_no in range(num_lanes - 1):
        lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
        lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
        for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
            dx = lane_markers_radius * np.cos(theta)
            dy = lane_markers_radius * np.sin(theta)
            w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))

    return w, cb

w, cb = environment_setup(w)

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(96,60), np.pi/2)
c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

# Level 2 (add this to the code)
c2 = Car(Point(world_width/2, 95), np.pi, 'yellow')
c2.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c2.velocity = Point(0, 3.0)
w.add(c2)

# Level 3 (add this to the code to level 2)
c3 = Car(Point(world_width/2, 24), 0, 'blue')
c3.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c3.velocity = Point(0, 3.0)
w.add(c3)

# Level 4 (add this to the code to level 3)
c4 = Car(Point(28,60), np.pi/2, 'green')
c4.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c4.velocity = Point(0, 3.0)
w.add(c4)

w.render() # This visualizes the world we just constructed.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    state_size = 4
    action_size = 15
    

    if args.test:
        try:
            agent = DQLAgent(state_size, action_size, epsilon=0)
            # Load the latest checkpoint
            checkpoint_path = "dql_agent_episode_500.pth"
            agent.model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded checkpoint: {checkpoint_path}")
            
            # Run test episodes
            for test_episode in range(10):  # Run 3 test episodes
                print(f"\nTest Episode {test_episode + 1}")
                
                # Reset environment
                c1.center = Point(96, 60)
                c1.heading = np.pi/2
                c1.velocity = Point(0, 3.0)
                
                # Level 2
                c2.max_speed = 30.0
                c2.velocity = Point(0, 3.0)
                w.add(c2)
                
                # Level 3
                c3.max_speed = 30.0
                c3.velocity = Point(0, 3.0)
                w.add(c3)

                # Level 4
                c4.center = Point(28,60)
                c4.heading = np.pi/2
                c4.velocity = Point(0, 3.0)
                w.add(c4)

                # Update other cars' positions
                radius = inner_building_radius + lane_width/2
                # Level 2
                update_car_position(c2, world_width/2, world_height/2, radius, 3.0, dt)   
                # Level 3
                update_car_position(c3, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
                # Level 4
                update_car_position(c4, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
                
                # Track performance metrics
                start_time = time.time()
                total_reward = 0
                lap_count = 0
                crossed_start = False
                
                while (time.time() - start_time) < 60:  # Run for 1 minute
                    # Get state and action
                    state = agent.get_state(c1, cb)
                    steering, throttle = agent.act(state) 
                    c1.set_control(steering, throttle)
                    
                    # Update other cars
                    radius = inner_building_radius + lane_width/2
                    # Level 2
                    update_car_position(c2, world_width/2, world_height/2, radius, 3.0, dt)
                    # Level 3
                    update_car_position(c3, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
                    # Level 4
                    update_car_position(c4, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
                    
                    # Track laps
                    if not crossed_start and abs(c1.center.x - 28) < lane_width and abs(c1.center.y - 60) < lane_width:
                        crossed_start = True
                    if crossed_start and abs(c1.center.x - 96) < lane_width and abs(c1.center.y - 60) < lane_width:
                        lap_count += 1
                        print(f"Lap completed: {lap_count}")
                        crossed_start = False
                        total_reward += 500
                    
                    # Update simulation
                    w.tick()
                    w.render()
                    
                    # Calculate reward
                    distance_to_center = c1.distanceTo(cb)
                    desired_distance = inner_building_radius + lane_width/2
                    distance_error = abs(distance_to_center - desired_distance)
                    
                    reward = 1.0 - distance_error * 0.1 - abs(steering) * 0.1
                    total_reward += reward
                    
                    # Check for collision
                    if w.collision_exists():
                        print("Collision detected!")
                        break
                
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Laps Completed: {lap_count}")
                time.sleep(1)  # Pause between episodes
                
            w.close()
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Training settings
        EPISODES = 10000
        batch_size = 32
        
        # Initialize DQL agent
        state_size = 4  # distance_to_center, velocity, heading_diff, heading
        action_size = 15  # 5 steering actions * 3 throttle actions
        agent = DQLAgent(state_size, action_size, epsilon=1.0)
        
        for episode in range(EPISODES):
            # Reset the environment
            c1.center = Point(96, 60)
            c1.heading = np.pi/2
            c1.velocity = Point(0, 3.0)

            # Level 2
            c2.center = Point(world_width/2, 95)
            c2.heading = np.pi
            c2.velocity = Point(0, 3.0)

            # Level 3
            c3.center = Point(world_width/2, 24)
            c3.heading = 0
            c3.velocity = Point(0, 3.0)

            # Level 4
            c4.center = Point(28,60)
            c4.heading = np.pi/2
            c4.velocity = Point(0, 3.0)
            
            # Add lap tracking variables
            last_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)
            lap_count = 0
            crossed_start = False
            
            total_reward = 0
            for time_step in range(1000):
                # Get current state
                state = agent.get_state(c1, cb)
                
                # Track laps
                # Calculate current angle relative to center
                current_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)
                
                # Check if car has crossed the start line (left side)
                if not crossed_start and abs(c1.center.x - 28) < lane_width and abs(c1.center.y - 60) < lane_width:
                    crossed_start = True
                
                # Check if car has crossed finish line (right side) after crossing start
                if crossed_start and abs(c1.center.x - 96) < lane_width and abs(c1.center.y - 60) < lane_width:
                    lap_count += 1
                    print(f"Lap completed: {lap_count}")
                    crossed_start = False  # Reset for next lap
                    reward += 500  # Bonus reward for completing a lap
                
                # Get action from agent
                steering, throttle = agent.act(state)
                c1.set_control(steering, throttle)
                
                # Update other cars' positions
                radius = inner_building_radius + lane_width/2
                update_car_position(c2, world_width/2, world_height/2, radius, 3.0, dt)   
                update_car_position(c3, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
                update_car_position(c4, world_width/2, world_height/2, radius + lane_width, 3.0, dt)

                # Advance simulation
                w.tick()
                w.render()
                
                # Calculate reward
                distance_to_center = c1.distanceTo(cb)
                desired_distance = inner_building_radius + lane_width/2
                distance_error = abs(distance_to_center - desired_distance)
                
                reward = 1.0  # Base reward for surviving
                reward -= distance_error * 0.1  # Penalty for being off-center
                reward -= abs(steering) * 0.1  # Small penalty for steering
                
                # Get new state
                next_state = agent.get_state(c1, cb)
                
                # Check if episode is done
                done = False
                if w.collision_exists():
                    reward = -1000  # Big penalty for collision
                    done = True
                
                # Store experience in memory
                action_idx = (agent.steering_actions.index(steering) * 
                             len(agent.throttle_actions) + 
                             agent.throttle_actions.index(throttle))
                agent.remember(state, action_idx, reward, next_state, done)
                
                # Train the network
                agent.replay(batch_size)
                
                total_reward += reward
                
                if done:
                    break

            # Update target network every episode
            agent.update_target_model()
            
            print(f"Episode: {episode + 1}/{EPISODES}, Score: {total_reward}")
            
            # Save the model periodically
            if episode % 100 == 0:
                torch.save(agent.model.state_dict(), f'dql_agent_episode_{episode}.pth')
        
        w.close()

else: # Let's use the keyboard input for human control
    from interactive_controllers import KeyboardController
    c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
    controller = KeyboardController(w)
    for k in range(600):
        c1.set_control(controller.steering, controller.throttle)
        
        # Update other cars' positions
        radius = inner_building_radius + lane_width/2
        update_car_position(c2, world_width/2, world_height/2, radius, 3.0, dt)   
        update_car_position(c3, world_width/2, world_height/2, radius + lane_width, 3.0, dt)
        update_car_position(c4, world_width/2, world_height/2, radius + lane_width, 3.0, dt)

        w.tick() # This ticks the world for one time step (dt second)
        w.render()
        time.sleep(dt/4) # Let's watch it 4x
        if w.collision_exists():
            import sys
            sys.exit(0)
    w.close()
