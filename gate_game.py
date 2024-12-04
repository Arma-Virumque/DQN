# File: gate_game.py
# Author: Michael Huelsman
# Copyright: Dr. Michael Andrew Huelsman 2024
# License: GNU GPLv3
# Created On: 06 Nov 2024
# Purpose:
# Notes:

import torch
import torch.nn as nn
from random import choices, choice, sample

class GateGameAgent(nn.Module):
    #Base class for a Deep Q-learning Agent for use with GateGame.
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, x):
        return torch.zeros((x.shape[0],len(GateGame.get_actions())))

    def training_step(self, experiences, target_net = None, sample_size: int = 256) -> None:
        #Run a single reinforcement training step to update the DQN.
        #:param experiences: A list of tuples containing: (state, action, reward, next_state, done)
        #:param target_net: An identical network to increase stability (a few updates old.)
        #:param sample_size: The size of the sample to train from (default: 256)
        #:return: None
        if len(experiences) < sample_size:
            return
        # Grab a random sampling of the experiences
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        exps = sample(experiences, sample_size)
        targets = self.__get_targets(exps, target_net)
        loss = self.loss_fn(*targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        #Returns the action the GateGameAgent would take for the given GateGame state.
        #:param state: A torch Tensor representing the current GateGame state.
        #:return: An integer indicating the action to take.
        state = torch.stack([state])
        return torch.argmax(self(state)[0]).item()

    def get_q(self, state):
        #Returns the Q values for the given state.
        #:param state: A torch Tensor representing the current GateGame state.
        #:return: A torch Tensor of the Q values for the state.
        state = torch.stack([state])
        return self(state)[0]

    def __get_targets(self, experiences, target_net, discount: float = 0.99):
        #Internal, private method of computing the target values for DQN loss calculation.
        #:param experiences: A list of tuples containing: (state, action, reward, next_state, done)
        #:param target_net: An identical network to increase stability (a few updates old.)
        #:param discount: The amount of reward discount applied to the next state's max Q value (default: 0.99)
        #:return: Tuple containing: Current Q values, target values for computing loss.
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float)
        q_vals = self(states).gather(1, actions).squeeze()
        with torch.no_grad():
            if target_net is None:
                next_q_vals = torch.max(self(next_states), dim=1)[0]
            else:
                next_q_vals = torch.max(target_net(next_states), dim=1)[0]
            targets = rewards + (discount * next_q_vals * (1 - dones))
        return q_vals, targets

class GateGame:
    #A class for a randomly generated game of selecting gates. Used for reinforcement learning
    # Game params
    __GAME_LENGTH = 30
    __START_POSITION = 2
    __GATES = 5
    __MAX_SCORE = 1
    # Actions
    __UP = 0
    __STAY = 1
    __DOWN = 2
    # Operations
    __NUM_OPS = 5
    __APPLY = 0
    __ADD = 1
    __MUL = 2
    __SUB = 3
    __DIV = 4
    # Magnitude Bounds
    __ADDITIVE_BOUNDS = (1, 10)
    __MULTIPLICATIVE_BOUNDS = (1, 3)
    __APPLICATIVE_BOUNDS = (1, 2)
    # Fog Settings
    __VIEW_DIST = 3
    # Operation Weights
    __OP_WEIGHTS = (10, 50, 20, 20, 10)

    def __init__(self):
        #Constructs a random instance of the Gate Game.
        # Setup plater
        self.player_position = GateGame.__START_POSITION
        self.__previous_positions = []
        self.player_score = 0
        self.current_gate = 0
        # Generate a random series of gates.
        self.__gates = []
        for _ in range(GateGame.__GAME_LENGTH):
            self.__gates.append([GateGame.__random_gate() for _ in range(GateGame.__GATES)])

    def reset(self):
        #Resets the game state to the starting state.
        self.player_position = GateGame.__START_POSITION
        self.__previous_positions = []
        self.current_gate = 0
        self.player_score = 0

    def step(self, action: int) -> int:
        #Steps forward the game by one action.
        #:param action: An integer among the options (0, 1, 2) indicating the actions (up, stay, down)
        #:return: The current score after applying the given action and the current gate.
        if self.current_gate >= GateGame.__GAME_LENGTH:
            return self.player_score
        self.player_position += (action-1)
        self.player_position %= GateGame.__GATES
        self.__previous_positions.append(self.player_position)
        self.__apply_gate(self.current_gate)
        self.current_gate += 1
        return self.player_score

    def random_action(self) -> int:
        #Returns a random action
        #:return: An integer indicating an action in the game.
        return choice(self.get_actions())

    def greedy_action(self) -> int:
        #Gets the next action to take based on the best gate to go through.
        #:return: The integer value of the action taken.
        self.__previous_positions.append(self.player_position)
        current_score = self.player_score
        current_position = self.player_position
        self.__apply_gate(self.current_gate)
        stay_value = self.player_score
        self.__previous_positions.pop()
        self.player_score = current_score
        self.player_position -= 1
        self.player_position %= GateGame.__GATES
        self.__previous_positions.append(self.player_position)
        self.__apply_gate(self.current_gate)
        up_value = self.player_score
        self.__previous_positions.pop()
        self.player_score = current_score
        self.player_position = current_position
        self.player_position += 1
        self.player_position %= GateGame.__GATES
        self.__previous_positions.append(self.player_position)
        self.__apply_gate(self.current_gate)
        down_value = self.player_score
        self.__previous_positions.pop()
        # Reset the position and score
        self.player_position = current_position
        self.player_score = current_score
        # Determine the winner
        if stay_value > up_value:
            if stay_value > down_value:
                return GateGame.__STAY
            return GateGame.__DOWN
        else:
            if up_value > down_value:
                return GateGame.__UP
            return GateGame.__DOWN

    def get_state(self) -> torch.Tensor:
        #Gets the current state of the game as a 1D torch Tensor object.
        #:return: A tensor representing the observable state.
        info_list = [self.player_position, self.player_score]
        for offset in range(self.__VIEW_DIST):
            gate_data = []
            if (self.current_gate + offset) in self.__view_range():
                for gate in self.__gates[self.current_gate+offset]:
                    gate_data.extend(gate)
            else:
                gate_data.extend([0 for i in range(2*GateGame.__GATES)])
            info_list.extend(gate_data)
        return torch.tensor(info_list, dtype=torch.float)

    def get_score(self) -> float:
        #Returns the current score.
        #:return: The player's score.
        return self.player_score/GateGame.__MAX_SCORE

    def is_complete(self) -> bool:
        #Queries the game to see if the game has been completed.
        #:return: Returns true if all gates have been passed.
        return self.current_gate >= GateGame.__GAME_LENGTH

    def __repr__(self):
        rep = f"Score: {self.player_score}\n"
        for gate_level in range(GateGame.__GATES):
            for gate_num in range(max(0, self.current_gate-2), self.__max_viewable_gate()+1):
                if gate_num == self.current_gate:
                    if gate_level == self.player_position:
                        rep += " X "
                    else:
                        rep += "   "
                gate = self.__gates[gate_num][gate_level]
                rep += f"{GateGame.__op_repr(gate[0])}{gate[1]}"
                rep += " "
            rep += "\n"
        return rep

    def __view_range(self):
        return range(self.current_gate, self.__max_viewable_gate()+1)

    def __max_viewable_gate(self):
        return min(self.current_gate+GateGame.__VIEW_DIST, GateGame.__GAME_LENGTH-1)



    #====================
    #   Private Methods
    #====================
    def __apply_gate(self, gate_num: int) -> None:
        #Applies the effect of going through the given gate.
        #:param gate_num: The gate to apply to the player's score.
        #:return: None
        if gate_num < 0:
            return
        gate_op, gate_val = self.__gates[gate_num][self.__previous_positions[gate_num]]
        match gate_op:
            case GateGame.__APPLY:
                self.__apply_gate(gate_num - gate_val)
            case GateGame.__ADD:
                self.player_score += gate_val
            case GateGame.__MUL:
                self.player_score *= gate_val
            case GateGame.__SUB:
                self.player_score -= gate_val
            case GateGame.__DIV:
                if self.player_score != 0:
                    self.player_score //= gate_val

    #====================
    #   Class Methods
    #====================
    @classmethod
    def get_actions(cls) -> tuple:
        return GateGame.__UP, GateGame.__STAY, GateGame.__DOWN

    @classmethod
    def __random_gate(cls):
        operation = choices(list(range(cls.__NUM_OPS)), cls.__OP_WEIGHTS)[0]
        value = 0
        if operation == 0:
            value = choice(list(range(*cls.__APPLICATIVE_BOUNDS)))
        elif operation%2 == 0:
            value = choice(list(range(*cls.__MULTIPLICATIVE_BOUNDS)))
        else:
            value = choice(list(range(*cls.__MULTIPLICATIVE_BOUNDS)))
        return operation, value

    @classmethod
    def __op_repr(cls, op):
        match op:
            case cls.__APPLY:
                return "A"
            case cls.__ADD:
                return "+"
            case cls.__MUL:
                return "*"
            case cls.__SUB:
                return "-"
            case cls.__DIV:
                return "/"
            case _:
                return "U"

def main():
    #Allows a user to play the game on the console, with a similar perspective to a game agent.
    welcome = """
    Welcome to the interactive version of GateGame!
    
    How GateGame works:
    --------------------
    Your character moves from the left to the right, one gate at a time.
    When you go through a gate the operation specified on the gate is used to modify your score.
    
    Gate Types:
    ------------
    + -> Add the specified number to the score.
    - -> Subtract the specified number to the score.
    * -> Multiply the score by the given number.
    \\ -> Divide the score by the given number (integer-division)
    A -> Apply the gate that you went through the given number of turns ago (recurs).
         Note: If there is no gate, score remains the same.
    
    Instructions:
    --------------
    As a player you have three possible actions at each step:
    0 - Move up by 1 gate.
    1 - Stay put.
    2 - Move down by 1 gate.
    Note: Actions are applied and then the character moves through a gate.
    
    Try to get the highest score possible! Good luck!
    """
    print(welcome)
    cont_playing = True
    while cont_playing:
        # Initialize the game
        game = GateGame()
        while not game.is_complete():
            # Display game state (partial view)
            print(game)
            valid_input = False
            action = 1
            while not valid_input:
                try:
                    action = int(input("Your move (0, 1, or 2):"))
                except ValueError:
                    print("Please enter an integer.")
                    action = -1
                if action in game.get_actions():
                    valid_input = True
            game.step(action)
        print(f"Final score: {game.get_score()}")
        play_again = input("Play again (y/n)?")
        if len(play_again) > 0 and play_again[0].lower() == 'n':
            cont_playing = False


if __name__ == '__main__':
    main()
