import pygame
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 560, 620  # Adjusted to match classic Pac-Man proportions
TILE_SIZE = 20  # Adjusted for better alignment with the original Pac-Man maze
ROWS = len("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Fixed row calculation
COLS = len("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # Fixed col calculation

# Colors
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man")

# Font for score display
font = pygame.font.Font(None, 36)
score = 0

# Original Pac-Man Level 1 Maze Layout (0 = Empty, 1 = Wall, 2 = Pellet, 3 = Power Pellet, 5 = Portal)
MAZE = [
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "X............XX............X",
    "X.XXXX.XXXXX.XX.XXXXX.XXXX.X",
    "X+XXXX.XXXXX.XX.XXXXX.XXXX+X",
    "X.XXXX.XXXXX.XX.XXXXX.XXXX.X",
    "X..........................X",
    "X.XXXX.XX.XXXXXXXXXX.XX.XXXX.X",
    "X.XXXX.XX.XXXXXXXXXX.XX.XXXX.X",
    "X......XX....XX....XX......X",
    "XXXXXX.XXXXX XX XXXXX.XXXXXX",
    "XXXXXX.XXXXX XX XXXXX.XXXXXX",
    "X      XX          XX      X",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "      .   XXXXXX .      ",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "X      XX          XX      X",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "XXXXXX.XX XXXXXX XX.XXXXXX",
    "X............XX............X",
    "X.XXXX.XXXXX.XX.XXXXX.XXXX.X",
    "X.XXXX.XXXXX.XX.XXXXX.XXXX.X",
    "X+...X................X...+X",
    "XXX.XX.XX.XXXXXXXXXX.XX.XX.XXX",
    "XXX.XX.XX.XXXXXXXXXX.XX.XX.XXX",
    "X......XX....XX....XX......X",
    "X.XXXXXXXXXX.XX.XXXXXXXXXX.X",
    "X.XXXXXXXXXX.XX.XXXXXXXXXX.X",
    "X............55............X",
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
]

# Convert character-based maze into numerical grid
num_maze = []
pellets = []
portal_positions = []
for row_idx, row in enumerate(MAZE):
    num_row = []
    for col_idx, cell in enumerate(row):
        if cell == "X":
            num_row.append(1)  # Wall
        elif cell == ".":
            num_row.append(2)  # Pellet
            pellets.append((col_idx * TILE_SIZE + TILE_SIZE // 2, row_idx * TILE_SIZE + TILE_SIZE // 2))
        elif cell == "+":
            num_row.append(3)  # Power Pellet
            pellets.append((col_idx * TILE_SIZE + TILE_SIZE // 2, row_idx * TILE_SIZE + TILE_SIZE // 2))
        elif cell == "5":
            num_row.append(5)  # Portal Area
            portal_positions.append((col_idx, row_idx))
        else:
            num_row.append(0)  # Open space
    num_maze.append(num_row)

# Load Pac-Man character
class Pacman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = TILE_SIZE // 5
        self.direction = "STOP"

    def can_move(self, dx, dy):
        new_x = (self.x + dx) // TILE_SIZE
        new_y = (self.y + dy) // TILE_SIZE
        
        # Prevent out-of-range errors
        if not (0 <= new_x < COLS and 0 <= new_y < ROWS):
            return False
        
        # Check collision with walls
        return num_maze[new_y][new_x] != 1

    def move(self):
        if self.direction == "UP" and self.can_move(0, -self.speed):
            self.y -= self.speed
        elif self.direction == "DOWN" and self.can_move(0, self.speed):
            self.y += self.speed
        elif self.direction == "LEFT" and self.can_move(-self.speed, 0):
            self.x -= self.speed
        elif self.direction == "RIGHT" and self.can_move(self.speed, 0):
            self.x += self.speed
        
        # Check for portal movement
        self.check_portal()
    
    def check_portal(self):
        pac_x, pac_y = self.x // TILE_SIZE, self.y // TILE_SIZE
        if (pac_x, pac_y) in portal_positions:
            if pac_x == 0:  # Left portal
                self.x = (COLS - 2) * TILE_SIZE
            elif pac_x == COLS - 1:  # Right portal
                self.x = TILE_SIZE
    
    def draw(self):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), TILE_SIZE // 2)

# Function to draw walls
def draw_walls():
    for row in range(len(num_maze)):
        for col in range(len(num_maze[row])):
            if num_maze[row][col] == 1:
                pygame.draw.rect(screen, BLUE, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

# Function to draw pellets
def draw_pellets():
    for pellet in pellets:
        pygame.draw.circle(screen, WHITE, pellet, 3)

# Function to draw score
def draw_score():
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# Create Pac-Man instance at a valid starting position
pacman = Pacman(TILE_SIZE + TILE_SIZE // 2, TILE_SIZE + TILE_SIZE // 2)

# Game loop
running = True
while running:
    screen.fill(BLACK)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                pacman.direction = "UP"
            elif event.key == pygame.K_DOWN:
                pacman.direction = "DOWN"
            elif event.key == pygame.K_LEFT:
                pacman.direction = "LEFT"
            elif event.key == pygame.K_RIGHT:
                pacman.direction = "RIGHT"
    
    # Update game state
    pacman.move()
    draw_walls()
    draw_pellets()
    pacman.draw()
    draw_score()
    
    # Refresh screen
    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()
sys.exit()
