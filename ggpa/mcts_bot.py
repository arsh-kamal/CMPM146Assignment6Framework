from __future__ import annotations
import math
import random
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose


class TreeNode:
    def __init__(self, param, parent=None):
        self.children = {}  # action.key() -> TreeNode
        self.parent = parent
        self.results = []  # List of rollout scores
        self.visits = 0  # Number of visits
        self.action = None  # Action that led to this node (None for root)
        self.param = param  # UCB-1 exploration parameter (c)
    
    def step(self, state):
        """Perform one MCTS iteration."""
        self.select(state)
        
    def get_best(self, state):
        """Return the action with the highest average score."""
        if not self.children:
            return random.choice(state.get_actions())
        
        best_action = None
        best_score = -float('inf')
        available_keys = [action.key() for action in state.get_actions()]
        for action_key, child in self.children.items():
            if action_key not in available_keys:
                continue  # Skip invalid actions
            if child.visits > 0:
                avg_score = sum(child.results) / child.visits
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = child.action
        if best_action is None:
            return random.choice(state.get_actions())
        return best_action
        
    def print_tree(self, indent=0):
        """Print the tree with scores and visit counts."""
        for action_key, child in self.children.items():
            avg_score = sum(child.results) / child.visits if child.visits > 0 else 0
            print("  " * indent + f"{action_key}: {avg_score:.2f} (visits: {child.visits})")
            child.print_tree(indent + 1)

    def select(self, state):
        """Select a node using UCB-1, then expand and rollout if needed."""
        available_actions = state.get_actions()
        available_keys = [action.key() for action in available_actions]
        
        # Expand if there are unexplored actions
        unexpanded = [action for action in available_actions if action.key() not in self.children]
        if unexpanded:
            self.expand(state, unexpanded)
            return
        
        # Select child using UCB-1, skipping invalid actions
        best_child = None
        best_ucb = -float('inf')
        total_visits = self.visits
        valid_children = [child for child in self.children.values() if child.action.key() in available_keys]
        
        if not valid_children:
            self.expand(state, available_actions)
            return
        
        for child in valid_children:
            if child.visits == 0:
                ucb = float('inf')  # Prioritize unvisited nodes
            else:
                avg_score = sum(child.results) / child.visits
                ucb = avg_score + self.param * math.sqrt(math.log(total_visits + 1) / child.visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        if best_child:
            new_state = state.copy_undeterministic()
            action_obj = best_child.action.to_action(new_state)
            if action_obj is None:
                print(f"WARNING: Action {best_child.action.key()} returned None in to_action")
                self.expand(new_state, new_state.get_actions())
                return
            try:
                new_state.step(best_child.action)
                best_child.select(new_state)
            except Exception as e:
                print(f"ERROR: Failed to apply action {best_child.action.key()}: {e}")
                self.expand(new_state, new_state.get_actions())

    def expand(self, state, available):
        """Expand by adding a new child for a random unexpanded action."""
        if not available:
            return  # No actions to expand
        action = random.choice(available)
        new_state = state.copy_undeterministic()
        action_obj = action.to_action(new_state)
        if action_obj is None:
            print(f"WARNING: Expand action {action.key()} returned None in to_action")
            return
        try:
            new_state.step(action)
        except Exception as e:
            print(f"ERROR: Failed to expand with action {action.key()}: {e}")
            return
        
        child = TreeNode(self.param, parent=self)
        child.action = action
        self.children[action.key()] = child
        
        score = child.rollout(new_state)
        child.backpropagate(score)

    def rollout(self, state):
        """Simulate random actions until game ends."""
        current_state = state.copy_undeterministic()
        while not current_state.ended():
            actions = current_state.get_actions()
            if not actions:
                break
            action = random.choice(actions)
            action_obj = action.to_action(current_state)
            if action_obj is None:
                print(f"WARNING: Rollout action {action.key()} returned None in to_action")
                continue
            try:
                current_state.step(action)
            except Exception as e:
                print(f"ERROR: Failed to apply rollout action {action.key()}: {e}")
                break
        return self.score(current_state)
        
    def backpropagate(self, result):
        """Record score and propagate to parent."""
        self.results.append(result)
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        
    def score(self, state):
        """Evaluate state: prioritize winning, penalize death."""
        if state.get_end_result() == -1:  # Player dead
            return -100  # Heavy penalty for losing
        base_score = state.score()  # Damage dealt to monster
        health_factor = state.health()  # Player health percentage
        return base_score + 0.2 * health_factor if base_score < 1 else 1.0


class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        super().__init__("MCTSAgent")
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            action_obj = actions[0].to_action(battle_state)
            if action_obj is None:
                print(f"WARNING: Single action {actions[0].key()} returned None in to_action")
                return EndAgentTurn()  # Fallback
            return action_obj
    
        t = TreeNode(self.param)
        for _ in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return EndAgentTurn()  # Fallback
        
        action_obj = best_action.to_action(battle_state)
        if action_obj is None:
            print(f"WARNING: Best action {best_action.key()} returned None in to_action")
            return EndAgentTurn()  # Fallback
        return action_obj
    
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]