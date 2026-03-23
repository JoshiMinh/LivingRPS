import pygame
import random
import time
from pygame.locals import K_UP, K_DOWN, K_ESCAPE, KEYDOWN, QUIT

import torch
from mover import Mover
from model import RPSAgentNet

# --- Initialization ---
pygame.mixer.init()
pygame.init()

# Set window icon and caption
try:
    pygame.display.set_icon(pygame.image.load("../icon.png"))
except Exception:
    pass
pygame.display.set_caption("LivingRPS")

# --- Constants ---
SCREEN_SIZE = 750
RADIUS = 15
PLAYER_COUNT = 15
MAX_VELO = 5
MAX_ACCEL = 2
FPS = 60

# --- Assets ---
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
font = pygame.font.Font(pygame.font.get_default_font(), 18)

rock_hit     = pygame.mixer.Sound("../assets/audio/rock.mp3")
scissors_hit = pygame.mixer.Sound("../assets/audio/scissors.mp3")
paper_hit    = pygame.mixer.Sound("../assets/audio/paper.mp3")

choices = [
    pygame.transform.scale(pygame.image.load("../assets/images/rock.png"),     (RADIUS * 2, RADIUS * 2)),
    pygame.transform.scale(pygame.image.load("../assets/images/scissors.png"), (RADIUS * 2, RADIUS * 2)),
    pygame.transform.scale(pygame.image.load("../assets/images/paper.png"),    (RADIUS * 2, RADIUS * 2)),
]

# --- Utility Functions ---
def clear():
    """Fill the screen with black."""
    screen.fill((0, 0, 0))

def create_players(player_count, screen_width, screen_height, max_velo, max_accel):
    """Create and return a list of Mover objects for each RPS type."""
    players = []
    quarter_h, quarter_w = screen_height // 4, screen_width // 4
    for t in range(3):
        for _ in range(player_count):
            players.append(Mover(
                t,
                (random.randint(quarter_h, 3 * quarter_h), random.randint(quarter_w, 3 * quarter_w)),
                (random.randint(-max_velo, max_velo), random.randint(-max_velo, max_velo)),
                (random.randint(-max_accel, max_accel), random.randint(-max_accel, max_accel)),
                screen_width, screen_height, max_velo, max_accel
            ))
    return players

def handle_collisions(players):
    """
    Handle collisions between players.
    Returns the count of each type after collisions.
    """
    totals = [0, 0, 0]
    for i, p1 in enumerate(players):
        p1_pos = p1.get_position()
        totals[p1.get_color()] += 1
        for j in range(i + 1, len(players)):
            p2 = players[j]
            p2_pos = p2.get_position()
            dist_sq = (p1_pos[0] - p2_pos[0])**2 + (p1_pos[1] - p2_pos[1])**2
            if dist_sq < (RADIUS * 2) ** 2:
                c1, c2 = p1.get_color(), p2.get_color()
                if c1 == c2:
                    continue
                # RPS rules: 0 beats 1, 1 beats 2, 2 beats 0
                if (c1, c2) in [(0, 1), (1, 2), (2, 0)]:
                    p2.update_color(c1)
                    [rock_hit, scissors_hit, paper_hit][c1].play()
                elif (c2, c1) in [(0, 1), (1, 2), (2, 0)]:
                    p1.update_color(c2)
                    [rock_hit, scissors_hit, paper_hit][c2].play()
    return totals

# --- Model Loading ---
model = RPSAgentNet()
try:
    model.load_state_dict(torch.load("../models/rps_agent.pth", map_location='cpu'))
except Exception:
    print("Warning: Could not load trained model. Using random weights.")
model.eval()

# --- Main Game Loop ---
def game_loop():
    wins = [0, 0, 0]
    going = True

    while going:
        players = create_players(PLAYER_COUNT, SCREEN_SIZE, SCREEN_SIZE, MAX_VELO, MAX_ACCEL)
        running = True

        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type in {QUIT, KEYDOWN} and (event.type == QUIT or event.key == K_ESCAPE):
                    return

            # --- AI Inference ---
            states = [p.get_state(players) for p in players]
            states_tensor = torch.tensor(states, dtype=torch.float)
            with torch.no_grad():
                q_values = model(states_tensor)
            actions = torch.argmax(q_values, dim=1).tolist()

            # --- Update Players ---
            for p, action in zip(players, actions):
                p.apply_action(action, players)
                p.update_physics()

            # --- Handle Collisions ---
            totals = handle_collisions(players)

            # --- Check Win Condition ---
            if any(t == PLAYER_COUNT * 3 for t in totals):
                wins[totals.index(max(totals))] += 1
                running = False

            # --- Drawing ---
            for player in players:
                screen.blit(choices[player.get_color()], player.get_position())
            pygame.display.flip()
            time.sleep(1 / FPS)
            clear()

        # --- Game Over Screen ---
        screen.blit(font.render('Game Over!', True, (255, 255, 255)), (75, 300))
        screen.blit(font.render(f'Rock: {wins[0]}  Scissors: {wins[1]}  Paper: {wins[2]}', True, (255, 255, 255)), (75, 375))
        screen.blit(font.render('Press ESC to Exit, Press any key to rerun', True, (255, 255, 255)), (75, 450))
        pygame.display.flip()

        # --- Wait for User Input ---
        waiting = True
        while waiting:
            event = pygame.event.wait()
            if event.type in {QUIT, KEYDOWN}:
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    going = False
                waiting = False

if __name__ == "__main__":
    game_loop()
    pygame.quit()