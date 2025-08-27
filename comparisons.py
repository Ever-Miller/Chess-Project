from problems import Board
from algorithms import *
from itertools import combinations
import random
import matplotlib.pyplot as plt

white = 0
black = 1
both = 2

AGENTS = [
  (MiniMaxAgentV1, "mm", 0),
  (MiniMaxAgentV6, "mm", 1),
  (MiniMaxAgentV7, "mm", 2),
  (MonteCarloAgentV1, "mc", 3),
  (MonteCarloAgentV2, "mc", 4),
  (RLAgentV2, "rl", 5)
]

# npy file setup
# [combo_index (15)][early game (0) or mid game (1)][game number]

with open(r"openings\eco_fens.txt", "r") as f:
  lines = [l.strip() for l in f if l.strip()]

# code to run the tournaments
def run_tournaments() -> None:
  all_vecs = PositionWeightVectors.load_all(model="V2")

  # get all combos of specified agents
  combos = list(combinations(AGENTS, 2))

  # initialize matrix
  games = np.load("tourney.npy")

  # go through each combo
  for i, ((w_agent_cls, w_flag, w_id),
            (b_agent_cls, b_flag, b_id)) in enumerate(combos):
    
    if i < 12:
      continue

    # games from opening position
    for n in range(100):
      # swap colors every game 
      if n % 2 == 0:
        white_agent, wflag, wid = (w_agent_cls(white), w_flag, w_id)
        black_agent, bflag, bid = (b_agent_cls(black), b_flag, b_id)
      else:
        white_agent, wflag, wid = (b_agent_cls(white), b_flag, b_id)
        black_agent, bflag, bid = (w_agent_cls(black), w_flag, w_id)

      board = Board()

      print(f"game {n}: opening  {white_agent.__class__.__name__}, {black_agent.__class__.__name__}")

      # play out the game
      while True:
        agent, flag = (white_agent, wflag) if board.side == white else (black_agent, bflag)
        if flag == "mm":
          agent.search_position(3, board)
        elif flag == "mc":
          agent.search_position(board)
        else:
          agent.search_position(3, board, all_vecs.weights)

        if agent.best_move == None:
          agent.best_move = 0

        if not board.make_move(agent.best_move, all_moves):
          break
      
      winner = board.winner
      is_tie = winner == 2

      winner_id = wid if winner == white else bid

      games[i][0][n] = 2 if is_tie else winner_id

    np.save("tourney.npy", games)

    # games from random position
    for n in range(100):
      # swap colors every game 
      if n % 2 == 0:
        white_agent, wflag, wid = (w_agent_cls(white), w_flag, w_id)
        black_agent, bflag, bid = (b_agent_cls(black), b_flag, b_id)
      else:
        white_agent, wflag, wid = (b_agent_cls(white), b_flag, b_id)
        black_agent, bflag, bid = (w_agent_cls(black), w_flag, w_id)
      fen = random.choice(lines)
      board = Board(fen=fen)

      print(f"game {n}: random  {white_agent.__class__.__name__}, {black_agent.__class__.__name__}")

      # play out the game
      while True:
        agent, flag = (white_agent, wflag) if board.side == white else (black_agent, bflag)
        if flag == "mm":
          agent.search_position(3, board)
        elif flag == "mc":
          agent.search_position(board)
        else:
          agent.search_position(3, board, all_vecs.weights)

        if agent.best_move == None:
          agent.best_move = 0

        if not board.make_move(agent.best_move, all_moves):
          break

      winner = board.winner
      is_tie = winner == 2

      winner_id = wid if winner == white else bid

      games[i][1][n] = 2 if is_tie else winner_id

    np.save("tourney.npy", games)

  
def show_tourney_results() -> None:
  games = np.load("tourney.npy")

  phase_names = ("opening position", "random position")
  agent_ids = [a[2] for a in AGENTS]
  agent_names = ["MiniMaxAgentV1", "MiniMaxAgentV6", "MiniMaxAgentV7", "MonteCarloAgentV1", "MonteCarloAgentV2", "RLAgentV2"]

  n = len(agent_ids)

  # build value arrays
  E = np.zeros((2, n, n))
  for combo_idx, (i,j) in enumerate(combinations(agent_ids, 2)):
      for phase in (0,1):
          res = games[combo_idx, phase]
          res = res[res != 0]  # drop unplayed
          if len(res)==0:
              continue
          total   = len(res)
          wins_i  = np.count_nonzero(res == i)
          wins_j  = np.count_nonzero(res == j)
          draws   = np.count_nonzero(res == 2)
          E[phase, i, j] = min(1, (wins_i + 0.5*draws) / total)
          E[phase, j, i] = min(1, (wins_j + 0.5*draws) / total)

  # plot
  fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
  for phase, ax in enumerate(axes):
      im = ax.imshow(E[phase], vmin=0, vmax=1, cmap='coolwarm')
      ax.set_title(phase_names[phase])
      ax.set_xticks(np.arange(n));  ax.set_yticks(np.arange(n))
      ax.set_xticklabels(agent_names, rotation=45, ha='right')
      ax.set_yticklabels(agent_names)
      # annotate
      for i in range(n):
          for j in range(n):
              val = E[phase,i,j]*100
              color = 'white' if abs(E[phase,i,j]-0.5)>0.25 else 'black'
              ax.text(j, i, f"{val:4.1f}%", ha='center', va='center', color=color, fontsize=8)

  # colorbar
  cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
  cbar.set_label("Expected score for row‑agent")

  plt.show()


if __name__ == "__main__":
  query = int(input("Run Tournament (0) or see results (1) ").strip())
  if query == 0:
    run_tournaments()
  else:
    show_tourney_results()
