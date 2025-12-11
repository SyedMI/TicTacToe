import os
import random
import numpy as np
from collections import deque

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        return self.get_state()

    def get_state(self):
        mapping = {"X": 1.0, "O": -1.0, " ": 0.0}
        return np.array([mapping[c] for c in self.board], dtype=np.float32)

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == " "]

    def step(self, action, player):
        if self.board[action] != " ":
            return self.get_state(), -10.0, True

        self.board[action] = player

        if self.check_win(player):
            return self.get_state(), (1.0 if player == "X" else -1.0), True

        if " " not in self.board:
            return self.get_state(), 0.0, True

        return self.get_state(), 0.0, False

    def check_win(self, p):
        wins = [(0,1,2),(3,4,5),(6,7,8),
                (0,3,6),(1,4,7),(2,5,8),
                (0,4,8),(2,4,6)]
        return any(all(self.board[i] == p for i in combo) for combo in wins)

    def render(self):
        b = self.board
        print()
        for i in range(0, 9, 3):
            print(f"{b[i]} | {b[i+1]} | {b[i+2]}")
        print()


def evaluate_board(board):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    for a,b,c in wins:
        line = [board[a], board[b], board[c]]
        if line.count("X") == 3:
            return 1
        if line.count("O") == 3:
            return -1
    if " " not in board:
        return 0
    return None

def minimax_score(board, player):
    res = evaluate_board(board)
    if res is not None:
        return res
    opponent = "O" if player == "X" else "X"
    scores = []
    for i in range(9):
        if board[i] == " ":
            newb = board.copy()
            newb[i] = player
            scores.append(minimax_score(newb, opponent))
    if player == "X":
        return max(scores)
    else:
        return min(scores)

def minimax_move(env, player="O"):
    best_move = None
    if player == "O":
        best_score = 999
        for m in env.available_actions():
            newb = env.board.copy()
            newb[m] = "O"
            score = minimax_score(newb, "X")
            if score < best_score:
                best_score = score
                best_move = m
    else:
        best_score = -999
        for m in env.available_actions():
            newb = env.board.copy()
            newb[m] = "X"
            score = minimax_score(newb, "O")
            if score > best_score:
                best_score = score
                best_move = m
    return best_move


class PolicyNetwork:
    def __init__(self, input_dim=9, hidden=64, output_dim=9, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0/(input_dim+hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, output_dim) * np.sqrt(2.0/(hidden+output_dim))
        self.b2 = np.zeros(output_dim)

    def forward(self, state):
        z1 = state.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        logits = a1.dot(self.W2) + self.b2
        cache = (state, z1, a1, logits)
        return logits, cache

    def get_action_and_logprob(self, state, avail_actions):
        logits, cache = self.forward(state)
        mask = np.full_like(logits, -1e9)
        for a in avail_actions:
            mask[a] = 0.0
        masked_logits = logits + mask
        exps = np.exp(masked_logits - np.max(masked_logits))
        probs = exps / (np.sum(exps) + 1e-12)
        action = np.random.choice(len(probs), p=probs)
        logprob = np.log(probs[action] + 1e-12)
        return action, logprob, probs, cache

    def compute_gradients(self, cache, probs, action, advantage):
        state, z1, a1, logits = cache
        d_logits = probs.copy()
        d_logits[action] -= 1.0
        d_logits *= advantage
        dW2 = np.outer(a1, d_logits)
        db2 = d_logits.copy()
        da1 = self.W2.dot(d_logits)
        dz1 = da1 * (1.0 - np.tanh(z1)**2)
        dW1 = np.outer(state, dz1)
        db1 = dz1.copy()
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_gradients(self, grads, lr=1e-3):
        self.W1 += lr * grads["W1"]
        self.b1 += lr * grads["b1"]
        self.W2 += lr * grads["W2"]
        self.b2 += lr * grads["b2"]


def discount_rewards(rewards, gamma):
    R = 0.0
    out = np.zeros_like(rewards, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        out[t] = R
    return out

def train(agent_net,
          env,
          episodes=20000,
          gamma=0.99,
          lr=1e-3,
          opponent="random",
          print_every=1000,
          baseline_alpha=0.01):
    baseline = 0.0
    baseline_window = deque(maxlen=1000)

    log = {"episode": [], "win": [], "draw": [], "loss": []}

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False

        states = []
        actions = []
        logprobs = []
        rewards = []
        caches = []
        probs_list = []

        while not done:
            avail = env.available_actions()
            if not avail:
                break
            action, logprob, probs, cache = agent_net.get_action_and_logprob(state, avail)
            next_state, reward, done = env.step(action, "X")

            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            caches.append(cache)
            probs_list.append(probs)
            rewards.append(reward)

            if done:
                break

            if opponent == "random":
                opp_action = random.choice(env.available_actions())
            elif opponent == "minimax":
                opp_action = minimax_move(env, player="O")
            else:
                opp_action = random.choice(env.available_actions())

            next_state2, reward2, done = env.step(opp_action, "O")
            rewards.append(reward2)
            state = next_state2

            if done:
                break

        returns = discount_rewards(rewards, gamma)
        agent_returns = returns[0::2]
        assert len(agent_returns) == len(actions)

        ep_return_mean = agent_returns.mean() if len(agent_returns) else 0.0
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * ep_return_mean
        baseline_window.append(ep_return_mean)

        total_grads = {"W1": np.zeros_like(agent_net.W1),
                       "b1": np.zeros_like(agent_net.b1),
                       "W2": np.zeros_like(agent_net.W2),
                       "b2": np.zeros_like(agent_net.b2)}

        for t in range(len(actions)):
            advantage = agent_returns[t] - baseline
            grads = agent_net.compute_gradients(caches[t], probs_list[t], actions[t], advantage)
            for k in total_grads:
                total_grads[k] += grads[k]

        agent_net.apply_gradients(total_grads, lr=lr)

        if any(r == 1.0 for r in rewards):
            w = 1
            d = 0
            l = 0
        elif any(r == -1.0 for r in rewards) or any(r == -10.0 for r in rewards):
            w = 0
            d = 0
            l = 1
        else:
            w = 0
            d = 1
            l = 0

        log["episode"].append(ep)
        log["win"].append(w)
        log["draw"].append(d)
        log["loss"].append(l)

        if ep % print_every == 0:
            lastN = 2000 if ep >= 2000 else ep
            wins = sum(log["win"][-lastN:])
            draws = sum(log["draw"][-lastN:])
            losses = sum(log["loss"][-lastN:])
            print(f"Ep {ep}/{episodes} | recent({lastN}) W/D/L: {wins}/{draws}/{losses} | baseline={baseline:.3f}")

    return log


def evaluate_policy(agent_net, env, n_games=1000, opponent="random"):
    wins = draws = losses = 0
    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            avail = env.available_actions()
            action, _, probs, cache = agent_net.get_action_and_logprob(state, avail)
            state, r, done = env.step(action, "X")
            if done:
                if r == 1.0:
                    wins += 1
                elif r == 0.0:
                    draws += 1
                elif r == -10.0:
                    losses += 1
                break

            if opponent == "random":
                opp_action = random.choice(env.available_actions())
            elif opponent == "minimax":
                opp_action = minimax_move(env, player="O")
            else:
                opp_action = random.choice(env.available_actions())
            state, r2, done = env.step(opp_action, "O")
            if done:
                if r2 == -1.0:
                    losses += 1
                elif r2 == 0.0:
                    draws += 1
                break
    return wins, draws, losses


if __name__ == "__main__":
    env = TicTacToe()
    policy = PolicyNetwork(hidden=64, seed=42)

    def sim_random_games(n=100):
        w=d=l=0
        for _ in range(n):
            s = env.reset()
            done=False
            while not done:
                a = random.choice(env.available_actions())
                s, r, done = env.step(a, "X")
                if done:
                    if r==1: w+=1
                    elif r==0: d+=1
                    else: l+=1
                    break
                a2 = random.choice(env.available_actions())
                s, r2, done = env.step(a2, "O")
                if done:
                    if r2==-1: l+=1
                    elif r2==0: d+=1
                    break
        print("Random-sim W/D/L:", w,d,l)

    sim_random_games(100)

    log = train(policy,
                env,
                episodes=8000,
                gamma=0.99,
                lr=5e-4,
                opponent="random",
                print_every=1000,
                baseline_alpha=0.02)

    # Evaluate vs random
    w,d,l = evaluate_policy(policy, env, n_games=500, opponent="random")
    print("Eval vs random (500): W/D/L =", w,d,l)

    # Evaluate vs minimax
    w,d,l = evaluate_policy(policy, env, n_games=500, opponent="minimax")
    print("Eval vs minimax (500): W/D/L =", w,d,l)

    print("\nPlay against trained policy! You are O (minimax will be used for opponent moves if you choose).")
    while True:
        env.reset()
        state = env.get_state()
        done = False
        env.render()
        print("You are O. Enter 0-8 to play. Enter -1 to quit.")
        while not done:
            try:
                mv = int(input("Your move (0-8): "))
            except:
                continue
            if mv == -1:
                raise SystemExit
            if mv not in env.available_actions():
                print("Invalid move, try again.")
                continue
            state, r, done = env.step(mv, "O")
            env.render()
            if done:
                if r == -1.0:
                    print("You (O) won!")
                elif r == 0.0:
                    print("Draw!")
                break

            ai_move, _, _, _ = policy.get_action_and_logprob(env.get_state(), env.available_actions())
            state, r, done = env.step(ai_move, "X")
            env.render()
            if done:
                if r == 1.0:
                    print("AI (X) won!")
                elif r == 0.0:
                    print("Draw!")
                break
        cont = input("Play again? (y/n): ").lower().strip()
        if cont != "y":
            break
