import pygame
import pygame.freetype
import random
import time
import mover
from pygame.locals import K_UP, K_DOWN, K_ESCAPE, KEYDOWN, QUIT

pygame.mixer.init()
pygame.init()

icon = pygame.image.load("images/icon.png")
pygame.display.set_icon(icon)
pygame.display.set_caption("LivingRPS")

def clear(window):
    window.fill((0, 0, 0))

def trim(v, upper, lower):
    return max(min(v, upper), lower)

def handle_bounds(x, y, x_bound, y_bound):
    return trim(x, x_bound, 0), trim(y, y_bound, 0)

move_speed = 3
screen_height = 750
screen_width = 750
screen_size = (screen_width, screen_height + screen_height // 10)
frames_per_second = 50
player_count = 20
max_velo = 5
max_accel = 2
radius = 15

rockHit = pygame.mixer.Sound("audio/rock.mp3")
scissorsHit = pygame.mixer.Sound("audio/scissors.mp3")
paperHit = pygame.mixer.Sound("audio/paper.mp3")

font = pygame.font.Font(pygame.font.get_default_font(), 18)

rock = pygame.image.load("images/rock.png")
scissors = pygame.image.load("images/scissors.png")
paper = pygame.image.load("images/paper.png")

rock = pygame.transform.scale(rock, (radius * 2, radius * 2))
paper = pygame.transform.scale(paper, (radius * 2, radius * 2))
scissors = pygame.transform.scale(scissors, (radius * 2, radius * 2))

choices = [rock, scissors, paper]
screen = pygame.display.set_mode([screen_width, screen_height])

wins = [0, 0, 0]
going = True

while going:
    players = []
    for t in range(3):
        for _ in range(player_count):
            coords = (random.randint(radius, screen_width - radius), random.randint(radius, screen_height - radius))
            velocity = (random.randint(-max_velo, max_velo), random.randint(-max_velo, max_velo))
            accel = (random.randint(-max_accel, max_accel), random.randint(-max_accel, max_accel))
            players.append(mover.Mover(t, coords, velocity, accel, screen_width, screen_height, max_velo, max_accel))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                going = False
                running = False

        visited = []
        positions = []
        totals = [0, 0, 0]

        for player in players:
            player.update_position()

        for player in players:
            for index in range(len(positions)):
                distance = (positions[index][0] - player.get_position()[0]) ** 2 + (positions[index][1] - player.get_position()[1]) ** 2
                if distance < (radius * 2) ** 2:
                    current_color = player.get_color()
                    collided_color = visited[index].get_color()

                    if current_color == 2 and collided_color == 1:
                        scissorsHit.play()
                        player.update_color(1)
                    elif current_color == 0 and collided_color == 2:
                        paperHit.play()
                        player.update_color(2)
                    elif current_color == 1 and collided_color == 0:
                        rockHit.play()
                        player.update_color(0)

                    visited[index].update_color(player.get_color())

            visited.append(player)
            positions.append(player.get_position())
            totals[player.get_color()] += 1

        for i in range(3):
            if totals[i] == player_count * 3:
                wins[i] += 1
                running = False
        
        players = visited

        for player in players:
            screen.blit(choices[player.get_color()], player.get_position())

        pygame.display.flip()
        time.sleep(1 / frames_per_second)
        clear(screen)

    # Game Over Screen
    clear(screen)
    screen.blit(font.render('Game Over!', True, (255, 255, 255)), (screen_width // 10, 4 * screen_height // 10))
    screen.blit(font.render(f'Rock: {wins[0]}  Paper: {wins[2]}  Scissors: {wins[1]}', True, (255, 255, 255)), (screen_width // 10, 5 * screen_height // 10))
    screen.blit(font.render('Press ESC to Exit, Press any key to restart', True, (255, 255, 255)), (screen_width // 10, 6 * screen_height // 10))
    pygame.display.flip()

    waiting = True
    while waiting:
        event = pygame.event.wait()
        if event.type == QUIT:
            going = False
            waiting = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                going = False
            waiting = False

pygame.quit()