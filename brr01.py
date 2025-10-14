import numpy as np
import random
import math

# --- Configuration for the Simplified Environment ---
# NOTE: The kinematics and physics in this environment are highly simplified
# for illustrative purposes only and do not represent real-world BVR dynamics.
# A real BVR environment requires a full 6DOF (degrees of freedom) simulation.

class BVRCombatEnv:
    """
    A highly simplified single-agent Reinforcement Learning environment
    for Beyond Visual Range (BVR) air combat simulation against a static enemy.

    State Space (Observed by the Agent):
    [0] Range (km)
    [1] Aspect Angle (radians, 0 is nose-on)
    [2] Closure Rate (m/s)
    [3] Own Fuel Level (0.0 to 1.0)
    [4] Enemy Missile Incoming (Binary: 0 or 1)
    """

    def __init__(self):
        # Environment Constants
        self.MAX_RANGE_KM = 100.0
        self.MISSILE_RANGE_KM = 30.0
        self.MAX_G = 9.0 # Placeholder for max maneuver
        self.BASE_SPEED_MPS = 300.0 # ~ Mach 0.9
        self.MAX_SIM_STEPS = 500

        # Action Space (Discrete)
        # 0: Maintain course/speed (Cruise)
        # 1: Turn Hard Left (Defensive/Offensive maneuver)
        # 2: Turn Hard Right (Defensive/Offensive maneuver)
        # 3: Fire Missile
        self.action_space_size = 4
        self.state_space_size = 5

        # Initialize state variables
        self.state = self._initial_state()
        self.current_step = 0
        self.max_fuel = 100.0

    def _initial_state(self):
        """Sets the initial state for both aircraft."""
        # Start at a random BVR range (50km to MAX_RANGE)
        range_km = random.uniform(50.0, self.MAX_RANGE_KM)
        aspect_rad = 0.0 # Start nose-on
        closure_rate = 50.0 # Initial closing at 50 m/s
        fuel = 1.0 # Normalized fuel
        enemy_missile_incoming = 0 # No missile fired yet

        return np.array([range_km, aspect_rad, closure_rate, fuel, enemy_missile_incoming])

    def reset(self):
        """Resets the environment to a new starting state."""
        self.state = self._initial_state()
        self.current_step = 0
        return self.state

    def step(self, action: int):
        """
        Takes one step in the environment based on the agent's action.

        Args:
            action (int): The chosen action (0-3).

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action not in range(self.action_space_size):
            raise ValueError("Invalid action.")

        current_range = self.state[0]
        current_aspect = self.state[1]
        current_closure = self.state[2]
        current_fuel = self.state[3]
        enemy_missile = self.state[4]

        # --- Transition Logic (Simplified Dynamics) ---
        new_range = current_range
        reward = 0.0
        done = False
        info = {}

        # 1. Update State based on Action
        time_delta = 1.0 # One second per step
        fuel_cost = 0.005 # Base fuel consumption

        if action == 0: # Cruise
            new_range -= (current_closure * time_delta) / 1000.0 # Update range in km
            pass # No major change in aspect/closure

        elif action == 1 or action == 2: # Hard Turn (Left/Right)
            # Simulates a high-G maneuver: burns more fuel, reduces closure, changes aspect
            fuel_cost = 0.015
            current_closure *= 0.98 # Maneuvering slows closure
            maneuver_angle = (15.0 * (action == 1) - 15.0 * (action == 2)) # Approx 15 deg change
            new_aspect = current_aspect + math.radians(maneuver_angle)
            self.state[1] = new_aspect % (2 * math.pi) # Normalize aspect angle
            reward -= 0.01 # Penalty for maneuvering (energy loss)

        elif action == 3: # Fire Missile
            # Can only fire if within missile range and missile not already fired
            if current_range <= self.MISSILE_RANGE_KM and not info.get('missile_fired', False):
                info['missile_fired'] = True
                reward += 1.0 # Reward for a successful launch condition
                # Assume enemy has a 50% chance of detecting the launch and firing back
                if random.random() < 0.5:
                    self.state[4] = 1 # Enemy missile is now incoming
            else:
                reward -= 0.5 # Large penalty for firing out of range/wasting missile

        # 2. Update Range & Closure Rate
        new_range -= (current_closure * time_delta) / 1000.0
        self.state[0] = max(0.0, new_range)

        # 3. Update Fuel
        self.state[3] = max(0.0, current_fuel - fuel_cost)

        # 4. Simple Enemy Missile Threat (If incoming)
        if enemy_missile == 1:
            # Threat persists for 50 steps (50 seconds)
            if self.current_step % 50 == 0:
                self.state[4] = 0 # Missile times out/is defeated
                reward += 0.5 # Small reward for surviving the threat
            else:
                reward -= 0.1 # Ongoing threat penalty

        # --- Termination Conditions ---
        self.current_step += 1

        # Check for successful kill (if a missile was fired)
        if 'missile_fired' in info and self.state[0] <= 1.0: # Close enough to assume hit/kill after firing
            reward += 100.0
            info['status'] = "WIN"
            done = True

        # Check for Fuel Out
        if self.state[3] <= 0.0:
            reward -= 50.0
            info['status'] = "OUT OF FUEL"
            done = True

        # Check for Max Steps
        if self.current_step >= self.MAX_SIM_STEPS:
            reward -= 10.0 # Penalty for not resolving the combat
            info['status'] = "TIME OUT"
            done = True

        # Check for Collision (Range too close without a kill)
        if self.state[0] <= 0.1 and not done:
            reward -= 200.0
            info['status'] = "COLLISION/LOST"
            done = True

        # 5. Calculate Reward Based on Objectives
        if not done:
            # Reward for closing the distance (primary BVR objective)
            reward += (current_range - self.state[0]) * 0.5

            # Reward for being in the 'Firing Envelope'
            if self.state[0] <= self.MISSILE_RANGE_KM:
                reward += 0.2
            elif self.state[0] < 5.0:
                reward -= 0.5 # Penalty for getting too close without resolution

        return self.state, reward, done, info

# --- Conceptual RL Agent (DQN Structure Placeholder) ---

class DQNAgent:
    """
    Conceptual Deep Q-Network Agent structure.
    In a real application, this would contain a neural network,
    experience replay, and target network logic.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # In a real DQN, this would be a prediction from the neural network:
        # return np.argmax(self.model.predict(state))
        
        # Placeholder: Always choose cruise/maneuver randomly when not exploring
        return random.choice([0, 1, 2])

    def learn(self, state, action, reward, next_state, done):
        """Placeholder for the learning process."""
        # In a real DQN, this would update the neural network based on the Bellman equation.
        if done:
            print(f"Agent finished episode with reward: {reward}")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        pass # Actual training logic goes here

# --- Simulation Execution ---

def run_bvr_simulation(episodes=100):
    """Main function to run the BVR combat simulation."""
    env = BVRCombatEnv()
    agent = DQNAgent(env.state_space_size, env.action_space_size)

    print("--- Starting Simplified BVR RL Training Simulation ---")
    
    # Simple example of what actions might look like for the first few episodes
    fixed_policy_actions = [3, 0, 0, 1, 0, 2, 0, 0, 3]

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_space_size])
        total_reward = 0
        
        for time in range(env.MAX_SIM_STEPS):
            # In a real setting, agent.choose_action(state) would be used.
            # Here, we use a simple placeholder logic for demonstration.
            if e < 5 and time < len(fixed_policy_actions):
                # Use a simple, non-learning policy for the first few steps
                action = fixed_policy_actions[time]
            else:
                # Use the conceptual agent for the rest
                action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_space_size])

            # In a real RL loop, we would store this transition in memory and learn from it:
            # agent.remember(state, action, reward, next_state, done)
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                status = info.get('status', 'Unknown')
                print(f"Episode: {e+1}/{episodes}, Steps: {time+1}, Status: {status.ljust(15)}, Final Range: {env.state[0]:.2f}km, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
                break
        
        # For simplicity, we break early after 10 episodes to show output faster
        if e >= 9:
            print("...Stopping simulation early for demonstration purposes...")
            break

if __name__ == "__main__":
    run_bvr_simulation()