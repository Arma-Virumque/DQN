# File: reinforcement_learning.py
# Author: Michael Dansereau
# Created On: 11 Nov 2024
# Purpose: create a reinforcement learning model to perform better in gate game than random or greedy actions

import torch
import torch.nn as nn
import random
import gate_game

class ReLe(gate_game.GateGameAgent):
    #Class for DQN agent
    def __init__(self, moves: list, operations: list, app_dist: int, row_gates: int):
        #:moves: list of all moves that can be made
        #:operations: list of all operations a gate can perform
        #:app_dist: the farthest back an applicator function can see
        #:rowGates: in number of gates in a given row
        super().__init__()
        self.moves = moves
        self.ops = operations
        self.app_dist = app_dist
        self.row_gates = row_gates

    def setApplicators(self, stateTensor: torch.tensor, action: int):
        #Returns tensor value of state with all applicators replaced with their actual value
        #:stateTensor: position, score, and upcoming gates in view distance as a tensor
        #:action: action the agent will take in the given state
        state = stateTensor.tolist()
        #pos holds the location the player will be after this next move
        pos = state[0]
        if action == 0:
            pos += 1
        elif action == 2:
            pos -= 1
        pos = int(pos)
        #operation and weight define what the applicators will become
        operation = state[pos]
        weight = state[pos + 1]
        
        #start represents the starting point of the for loop
        start = 2 * self.row_gates + 2
        #end represents the end point of the for loop
        end = 4 * self.row_gates + 2
        #for loop extends as far as applicators can view and changes their values
        for x in range(self.app_dist):
            for i in range(start, end, 2):
                #print(i)
                if state[i] == self.ops[0] and state[i + 1] == 1:
                    state[i] = operation
                    state[i + 1] = weight
            start = end
            end += (2 * self.row_gates)
        return torch.tensor(state)

    def reward(self, stateTensor: torch.tensor, action: int):
        #Returns an int value representing the reward given for a particular move
        state = self.setApplicators(stateTensor, action).tolist()
        #score holds the current score
        score = state[1]
        #pos holds the location the player will be after this next move
        pos = state[0]
        if action == 0:
            pos += 1
        elif action == 2:
            pos -= 1
        pos = int(pos)
        #operation and weight are the properties of upcoming gate
        operation = state[2 * pos + 2]
        weight = state[2 * pos + 3]

        #new_score represents the score after the action has been taken
        new_score = score
        if operation == 1:
            return weight
        elif operation == 2:
            new_score = new_score * weight
        elif operation == 3:
            return 0 - weight
        elif operation == 4:
            new_score = new_score / weight
        return new_score - score

def main():
    game = gate_game.GateGame()
    agent = ReLe([0, 1, 2], [0, 1, 2, 3, 4, 5], 2, 5)
    #number of training steps
    num_steps = 10
    #number of games per training step
    num_games = 120
    #chance of a game being repeated (repeated games count towards num_games)
    games_repeat = 1.0
    #rate at which the chance of repetition declines
    decay = 0.02
    #training loop for reinforcement learner
    for x in range(num_steps):
        #experiences list
        experiences = []
        for i in range(num_games):
            #while loop completes a single game
            while not game.is_complete():
                action = agent.get_action(game.get_state())
                state = agent.setApplicators(game.get_state(), action)
                reward = agent.reward(state, action)
                #executes the forward function on the neural net
                agent(state)
                game.step(action)
                next_state = agent.setApplicators(game.get_state(), agent.get_action(game.get_state()))
                done = game.is_complete()
                experiences.append((state, action, reward, next_state, done))
            #either repeats the same game again or creates a new one
            if random.random() < games_repeat:
                game.reset()
            else:
                game = gate_game.GateGame()
            games_repeat = games_repeat - (games_repeat * decay)
        #this is supposed to train the neural network but instead it just crashes
        #and I have absolutely no clue why.
        agent.training_step(experiences)
            
    #Testing loop
    num_tests = 10
    #cumulative scores for greedy, random, and agent across all testing games
    agentScore = 0.0
    randomScore = 0.0
    greedyScore = 0.0
    #testing loop
    for i in range(num_tests):
        game = gate_game.GateGame()
        #agent game
        while not game.is_complete():
            game.step(agent.get_action(game.get_state()))
        agentScore += game.get_score()
        game.reset()
        #greedy game
        while not game.is_complete():
            game.step(game.greedy_action())
        greedyScore += game.get_score()
        game.reset()
        #random game
        while not game.is_complete():
            game.step(game.random_action())
        randomScore += game.get_score()
    #print results
    print(f"Games Played: {num_tests}\nAverage Agent Score: {(agentScore / num_tests):.2f}\nAverage Greedy Score: {(greedyScore / num_tests):.2f}\nAverage Random Score: {(randomScore / num_tests):.2f}")
if __name__ == '__main__':
    main()
