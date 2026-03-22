<h1 align="center">
  <img src="assets/images/icon.png" alt="LivingRPS Icon" width="60"><br>
  LivingRPS
</h1>

<p align="center">
  AI-driven Rock-Paper-Scissors simulation — every entity on screen is a DQN agent hunting its prey and fleeing its predator.
</p>

<p align="center">
  <img src="demo.png" alt="Game Demo" width="600">
</p>

---

## Requirements

- Python 3.9+
- `numpy`, `pygame`, `torch`

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Running

**One-click (Windows):** double-click `run.bat` — it checks for Python, installs dependencies if needed, trains the model if weights are missing, then launches the game.

**Manual:**

```bash
python main.py
```

---

## Training (optional)

The pre-trained model is included in `models/rps_agent.pth`. To retrain from scratch:

```bash
python train.py
```

Training runs 1 000 DQN episodes and saves the new weights to `models/rps_agent.pth`.

---

## Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `↑` | Print entity states to console |
| `↓` | Resume (after `↑`) |
| Any key (Game Over screen) | Restart |

---

## Project Structure

```
LivingRPS/
├── assets/
│   ├── audio/          # Sound effects
│   └── images/         # Sprites & icon
├── models/             # Trained model weights
├── src/
│   ├── model.py        # RPSAgentNet (DQN architecture)
│   └── mover.py        # Entity physics & AI decision logic
├── main.py             # Game loop
├── train.py            # DQN training script
├── requirements.txt
└── run.bat             # One-click launcher (Windows)
```

---

> Fork of [ethan-schaffer/LivingRPS](https://github.com/ethan-schaffer/LivingRPS) with AI enhancements and structural improvements.