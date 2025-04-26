import pygame
import random
import time
import mover
import torch
from pygame.locals import K_UP, K_DOWN, K_ESCAPE, KEYDOWN, QUIT
from rps_agent_model import RPSAgentNet

pygame.mixer.init()
pygame.init()

pygame.display.set_icon(pygame.image.load("images/icon.png"))
pygame.display.set_caption("LivingRPS")
screen = pygame.display.set_mode((750, 750))
font = pygame.font.Font(pygame.font.get_default_font(), 18)

rock_hit = pygame.mixer.Sound("audio/rock.mp3")
scissors_hit = pygame.mixer.Sound("audio/scissors.mp3")
paper_hit = pygame.mixer.Sound("audio/paper.mp3")
radius = 15
choices = [
    pygame.transform.scale(pygame.image.load("images/rock.png"), (radius * 2, radius * 2)),
    pygame.transform.scale(pygame.image.load("images/scissors.png"), (radius * 2, radius * 2)),
    pygame.transform.scale(pygame.image.load("images/paper.png"), (radius * 2, radius * 2)),
]

def clear():
    screen.fill((0, 0, 0))

def create_players(player_count, screen_width, screen_height, max_velo, max_accel):
    players = []
    quarter_h, quarter_w = screen_height // 4, screen_width // 4
    for t in range(3):
        for _ in range(player_count):
            players.append(mover.Mover(
                t,
                (random.randint(quarter_h, 3 * quarter_h), random.randint(quarter_w, 3 * quarter_w)),
                (random.randint(-max_velo, max_velo), random.randint(-max_velo, max_velo)),
                (random.randint(-max_accel, max_accel), random.randint(-max_accel, max_accel)),
                screen_width, screen_height, max_velo, max_accel
            ))
    return players

def handle_collisions(players, visited, positions, totals):
    for player in players:
        for i, pos in enumerate(positions):
            if (pos[0] - player.get_position()[0]) ** 2 + (pos[1] - player.get_position()[1]) ** 2 < (radius * 2) ** 2:
                current, collided = player.get_color(), visited[i].get_color()
                if current == 2 and collided == 1:
                    scissors_hit.play()
                    player.update_color(1)
                elif current == 0 and collided == 2:
                    paper_hit.play()
                    player.update_color(2)
                elif current == 1 and collided == 0:
                    rock_hit.play()
                    player.update_color(0)
                visited[i].update_color(player.get_color())
        visited.append(player)
        positions.append(player.get_position())
        totals[player.get_color()] += 1
    return visited, totals

model = RPSAgentNet()
model.load_state_dict(torch.load("rps_agent_model.pth"))
model.eval()

def game_loop():
    player_count, max_velo, max_accel, fps = 15, 5, 2, 60
    wins = [0, 0, 0]
    going = True
    
    while going:
        players = create_players(player_count, 750, 750, max_velo, max_accel)
        for p in players:
            p.ai_model = model
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type in {QUIT, KEYDOWN} and (event.type == QUIT or event.key == K_ESCAPE):
                    return
                if event.type == KEYDOWN and event.key == K_UP:
                    print(*players, sep="\n")
                    while any(e.type != KEYDOWN or e.key != K_DOWN for e in pygame.event.get()):
                        pass
            
            visited, positions, totals = [], [], [0, 0, 0]
            for player in players:
                player.update_position(players)
            players, totals = handle_collisions(players, visited, positions, totals)
            
            if any(totals[i] == player_count * 3 for i in range(3)):
                wins[totals.index(max(totals))] += 1
                running = False
            
            for player in players:
                screen.blit(choices[player.get_color()], player.get_position())
            pygame.display.flip()
            time.sleep(1 / fps)
            clear()
        
        screen.blit(font.render('Game Over!', True, (255, 255, 255)), (75, 300))
        screen.blit(font.render(f'Rock: {wins[0]} Paper: {wins[2]} Scissors: {wins[1]}', True, (255, 255, 255)), (75, 375))
        screen.blit(font.render('Press ESC to Exit, Press any key to rerun', True, (255, 255, 255)), (75, 450))
        pygame.display.flip()
        
        while True:
            event = pygame.event.wait()
            if event.type in {QUIT, KEYDOWN}:
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    going = False
                break

game_loop()
pygame.quit()