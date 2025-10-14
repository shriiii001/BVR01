import numpy as np
import random
import math

# ==============================================================================
# 1. THE ENVIRONMENT (BVRCombatEnv)
# The environment defines the world, the rules, and the physics (simplified).
# ==============================================================================

class BVRCombatEnv:
    """
    A simplified single-agent Reinforcement Learning environment for BVR combat.
    The agent learns the optimal policy (what to do and when) to defeat the enemy.
    """

    def __init__(self):
        # --- Environment Constants ---
        self.MAX_RANGE_KM = 100.0
        self.MISSILE_RANGE_KM = 30.0 # Critical threshold for firing
        self.MAX_SIM_STEPS = 500

        # --- State Space Definition (What the Agent Observes) ---
        # This is the "S" (State) in the RL equation.
        # Index 0: Range (km)
        # Index 1: Aspect Angle (radians, 0 is nose-on)
        # Index 2: Closure Rate (m/s)
        # Index 3: Own Fuel Level (0.0 to 1.0)
        # Index 4: Enemy Missile Incoming (Binary: 0=No, 1=Yes)
        self.state_space_size = 5

        # --- Action Space Definition (What the Agent Can Do) ---
        # This is the "A" (Action) in the RL equation.
        # 0: Cruise/Maintain Course (Fuel efficient, steady closure)
        # 1: Turn Hard Left (Defensive/Offensive maneuver)
        # 2: Turn Hard Right (Defensive/Offensive maneuver)
        # 3: Fire Missile (High reward potential, but risk of waste)
        self.action_space_size = 4

        self.state = self._initial_state()
        self.current_step = 0

    def _initial_state(self):
        """Sets the initial randomized state for a new combat episode."""
        range_km = random.uniform(50.0, self.MAX_RANGE_KM)
        aspect_rad = 0.0
        closure_rate = 50.0
        fuel = 1.0
        enemy_missile_incoming = 0
        return np.array([range_km, aspect_rad, closure_rate, fuel, enemy_missile_incoming])

    def reset(self):
        """Resets the environment for a new training episode."""
        self.state = self._initial_state()
        self.current_step = 0
        return self.state

    def step(self, action: int):
        """
        The core transition function. Executes an action and returns S', R, D.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Unpack the current state
        current_range, current_aspect, current_closure, current_fuel, enemy_missile = self.state

        reward = 0.0 # R (Reward)
        done = False # D (Done/Termination)
        info = {}

        # --- Simplified Physics and Action Costs ---
        time_delta = 1.0 # Each step is 1 second
        fuel_cost = 0.005 # Base fuel consumption

        if action == 0: # Cruise
            pass # Simple range closure
        elif action == 1 or action == 2: # Hard Turn
            fuel_cost = 0.015
            current_closure *= 0.98 # Maneuvering reduces effective closure
            reward -= 0.01 # Penalty: Maneuvering costs energy/reward
        elif action == 3: # Fire Missile
            if current_range <= self.MISSILE_RANGE_KM:
                info['missile_fired'] = True
                reward += 1.0 # Initial reward for firing successfully
                # Simulate a random counter-fire chance
                if random.random() < 0.5:
                    enemy_missile = 1
            else:
                reward -= 0.5 # Penalty for wasting a missile out of range

        # 1. Range Update (Transition S -> S')
        new_range = max(0.0, current_range - (current_closure * time_delta) / 1000.0)
        self.state[0] = new_range

        # 2. Fuel Update
        self.state[3] = max(0.0, current_fuel - fuel_cost)

        # 3. Enemy Threat Handling
        if enemy_missile == 1:
            reward -= 0.1 # Ongoing threat penalty
            if self.current_step % 50 == 0: # Threat is defeated after 50 seconds
                enemy_missile = 0
                reward += 0.5 # Reward for surviving the threat
        self.state[4] = enemy_missile

        # --- Termination Conditions (Game Over) ---
        self.current_step += 1

        if info.get('missile_fired') and self.state[0] <= 1.0:
            reward += 100.0 # Huge positive reward for a WIN
            info['status'] = "WIN"
            done = True
        elif self.state[3] <= 0.0:
            reward -= 50.0 # Large negative reward for running OUT OF FUEL
            info['status'] = "OUT OF FUEL"
            done = True
        elif self.state[0] <= 0.1 and not done:
            reward -= 200.0 # Massive penalty for COLLISION/LOST
            info['status'] = "COLLISION/LOST"
            done = True
        elif self.current_step >= self.MAX_SIM_STEPS:
            reward -= 10.0
            info['status'] = "TIME OUT"
            done = True

        # 4. Continuous Reward (Encourages closing distance)
        if not done:
            reward += (current_range - self.state[0]) * 0.5
            if self.state[0] <= self.MISSILE_RANGE_KM:
                reward += 0.2 # Small bonus for being in the fire envelope

        # Return the next state, reward, done signal, and debug info
        return self.state, reward, done, info

# ==============================================================================
# 2. THE AGENT (DQNAgent - Conceptual)
# This is the "brain" that learns the Q-Values.
# ==============================================================================

class DQNAgent:
    """
    Conceptual Agent Structure. In a full implementation, this would use a
    Neural Network to learn the Q-values for every State-Action pair.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95    # Discount Factor: How much future rewards matter
        self.epsilon = 1.0   # Exploration Rate (Starts at 100% exploration)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # NOTE: A real DQN agent would initialize its Neural Network model here.

    def choose_action(self, state):
        """
        Implements the Epsilon-Greedy strategy:
        Explore (random action) or Exploit (best learned action).
        """
        if random.random() <= self.epsilon:
            # EXPLORE: Choose a random action (0 to 3)
            return random.randrange(self.action_size)
        
        # EXPLOIT: In a real model, you'd predict the best action from the NN.
        # For this demonstration, we use a placeholder policy:
        current_range = state[0]
        if current_range <= 25.0:
            return 3 # Fire!
        
        return 0 # Cruise to close distance

    def learn(self, state, action, reward, next_state, done):
        """
        Placeholder for the core RL learning step based on the Bellman equation.
        Updates the agent's knowledge and decays the exploration rate.
        """
        # 1. Update Epsilon (less random exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 2. Actual Learning (In a real DQN, this is where the NN would train)
        # Target Q = Reward + gamma * max(Q(S', A'))
        # Error = Target Q - Current Q(S, A)
        # Backpropagate the error through the network.
        pass

# ==============================================================================
# 3. THE TRAINING LOOP (The Practical Demonstration)
# This simulates the interaction between the Agent and the Environment.
# ==============================================================================

def run_rl_training(episodes=500):
    """
    Simulates the training process for the BVR RL agent.
    """
    env = BVRCombatEnv()
    agent = DQNAgent(env.state_space_size, env.action_space_size)

    print("--- BVR Combat RL Training Simulation Start ---")
    print(f"Goal: Learn the optimal State-Action sequence to maximize reward.")

    for e in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        
        for time in range(env.MAX_SIM_STEPS):
            # Agent selects an action based on the current state (Epsilon-Greedy)
            action = agent.choose_action(state)

            # Environment executes the action and returns S', R, D
            next_state, reward, done, info = env.step(action)
            
            # Agent uses the result to improve its policy
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                status = info.get('status', 'Unknown')
                print(f"Episode: {e}/{episodes} | Steps: {time+1} | Status: {status.ljust(15)} | Final Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
                break
        
        # Show progress less frequently after the first few episodes
        if e % 50 == 0:
             print("-" * 60)


if __name__ == "__main__":
    run_rl_training(episodes=250)
