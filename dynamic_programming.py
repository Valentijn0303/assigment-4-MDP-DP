#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self, env, gamma=1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)

        while True: # repeat until convergence
            delta = 0 # will store the maximum change in value function
            for s in range(env.n_states): # loop over all states
                v = V_s[s] # store the old value of the state#
                action_values = np.zeros(env.n_actions) # will store the action values for all actions in state s
                for a in range(env.n_actions): # loop over all actions
                    s_prime, reward = env.transition_function(s, env.actions[a]) # get the next state and reward
                    action_values[a] = reward + gamma * V_s[s_prime] # compute the action value
                V_s[s] = np.max(action_values)  # update the value of the state
                delta = max(delta, np.abs(v - V_s[s])) # update the maximum change in value function
            print(f'Error: {delta}') # print the maximum change in value function
            if delta < theta: # check for convergence
                break # if converged, break the loop

        self.V_s = V_s # store the value table
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        while True: # repeat until convergence
            delta = 0 # will store the maximum change in value function
            for s in range(env.n_states): # loop over all states
                for a in range(env.n_actions): # loop over all actions
                    q = Q_sa[s, a] # store the old value of the state-action pair
                    s_prime, reward = env.transition_function(s, env.actions[a]) # get the next state and reward
                    Q_sa[s, a] = reward + gamma * np.max(Q_sa[s_prime]) # compute the action value
                    delta = max(delta, np.abs(q - Q_sa[s, a])) # update the maximum change in value function
            print(f'Error: {delta}') # print the maximum change in value function
            if delta < theta: # check for convergence
                break # if converged, break the loop

        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None: # if a value table is available, use it to compute the action values
                action_values = np.zeros(env.n_actions) # will store the action values for all actions in state s
                for a in range(env.n_actions): # loop over all actions
                    s_prime, reward = env.transition_function(current_state, available_actions[a]) # get the next state and reward
                    action_values[a] = reward + self.V_s[s_prime] # compute the action value
                greedy_action = available_actions[np.argmax(action_values)] # get the greedy action

            elif table == 'Q' and self.Q_sa is not None: 
                action_values = self.Q_sa[current_state]
                greedy_action = env.actions[np.argmax(action_values)]
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None

            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

