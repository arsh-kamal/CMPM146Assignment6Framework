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
        self.children = {}
        self.parent = parent
        self.results = []
        self.visits = 0
        self.action = None
        self.param = param

    def step(self, state):
        self.select(state)

    def get_best(self, state):
        if not self.children:
            return random.choice(state.get_actions())

        best_action = None
        best_score = -float('inf')
        available_keys = [action.key() for action in state.get_actions()]
        for action_key, child in self.children.items():
            if action_key not in available_keys:
                continue
            if child.visits > 0:
                avg_score = sum(child.results) / child.visits
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = child.action
        return best_action or random.choice(state.get_actions())

    def print_tree(self, indent=0):
        for action_key, child in self.children.items():
            avg_score = sum(child.results) / child.visits if child.visits > 0 else 0
            print("  " * indent + f"{action_key}: {avg_score:.2f} (visits: {child.visits})")
            child.print_tree(indent + 1)

    def select(self, state):
        available_actions = state.get_actions()
        available_keys = [action.key() for action in available_actions]

        # Expand if untried actions exist
        unexpanded = [action for action in available_actions if action.key() not in self.children]
        if unexpanded:
            self.expand(state, unexpanded)
            return

        # Choose child via UCB-1
        best_child = None
        best_ucb = -float('inf')
        total_visits = self.visits
        valid_children = [child for child in self.children.values() if child.action.key() in available_keys]

        if not valid_children:
            self.expand(state, available_actions)
            return

        for child in valid_children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                avg_score = sum(child.results) / child.visits
                ucb = avg_score + self.param * math.sqrt(math.log(total_visits + 1) / child.visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        if best_child:
            new_state = state.copy_undeterministic()
            try:
                new_state.step(best_child.action)
                best_child.select(new_state)
            except Exception:
                return

    def expand(self, state, available):
        if not available:
            return
        action = random.choice(available)
        new_state = state.copy_undeterministic()
        try:
            new_state.step(action)
        except Exception:
            return

        child = TreeNode(self.param, parent=self)
        child.action = action
        self.children[action.key()] = child

        score = child.rollout(new_state)
        child.backpropagate(score)

    def rollout(self, state):
        """Simulate until end, using a biased random policy."""
        current_state = state.copy_undeterministic()
        while not current_state.ended():
            actions = current_state.get_actions()
            if not actions:
                break

            # Prefer block/low-cost/energy cards if available
            def action_priority(a):
                key = a.key().lower()
                if "offering" in key:
                    return 3  # deprioritize unless desperate
                if "defend" in key or "shrug" in key or "seeingred" in key:
                    return 0  # prioritize block/energy
                if "strike" in key or "thunderclap" in key:
                    return 1
                return 2
                return 1  # middle priority

            actions.sort(key=action_priority)
            action = actions[0]

            try:
                current_state.step(action)
            except Exception:
                break
        return self.score(current_state)

    def backpropagate(self, result):
        self.results.append(result)
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)

    def score(self, state):
        if state.get_end_result() == -1:
            return 0  # Loss
        base = state.score()
        health = state.health()
        
        # If score is close to 1 but health is low, reduce overall score slightly
        return base * (0.6 + 0.4 * health)


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
            return action_obj if action_obj else EndAgentTurn()

        t = TreeNode(self.param)
        for _ in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)

        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()

        if not best_action:
            return EndAgentTurn()
        action_obj = best_action.to_action(battle_state)
        return action_obj if action_obj else EndAgentTurn()

    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]

    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
