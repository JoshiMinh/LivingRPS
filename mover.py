import random
import torch

# RGB color definitions
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# Utility function to constrain a value within a range
def trim(v, upper, lower):
    return max(lower, min(v, upper))

class Mover:
    def __init__(self, thing, coords, v, a, max_height, max_width, max_velo, max_accel, ai_model=None):
        self.ticks = 0
        self.time = 0
        self.color = thing  # Represents the type of the mover
        self.coord_x, self.coord_y = coords
        self.velocity_x, self.velocity_y = v
        self.acceleration_x, self.acceleration_y = a
        self.max_height = max_height
        self.max_width = max_width
        self.max_velo = max_velo
        self.max_accel = max_accel
        self.ai_model = ai_model

    def get_color(self):
        return self.color

    # Update color based on game rules
    def update_color(self, c):
        if (c, self.color) in [(0, 1), (1, 2), (2, 0)]:
            self.color = c

    def get_position(self):
        return self.coord_x, self.coord_y

    # Decide acceleration based on AI model and nearby entities
    def decide_acceleration(self, others):
        my_type = self.color
        threat_type, prey_type = (2, 1) if my_type == 0 else (0, 2) if my_type == 1 else (1, 0)

        nearest_threat, nearest_prey = None, None
        min_d_threat, min_d_prey = float('inf'), float('inf')
        px, py = self.coord_x, self.coord_y

        # Find nearest threat and prey
        for other in others:
            if other is self:
                continue
            ox, oy = other.get_position()
            d2 = (ox - px)**2 + (oy - py)**2
            if other.get_color() == threat_type and d2 < min_d_threat:
                min_d_threat, nearest_threat = d2, (ox, oy)
            if other.get_color() == prey_type and d2 < min_d_prey:
                min_d_prey, nearest_prey = d2, (ox, oy)

        # Normalize distances and directions
        dx_t, dy_t, dist_t = 0.0, 0.0, 1.0
        dx_p, dy_p, dist_p = 0.0, 0.0, 1.0
        if nearest_threat:
            dx_t = (nearest_threat[0] - px) / self.max_width
            dy_t = (nearest_threat[1] - py) / self.max_height
            dist_t = ((dx_t**2 + dy_t**2)**0.5)
        if nearest_prey:
            dx_p = (nearest_prey[0] - px) / self.max_width
            dy_p = (nearest_prey[1] - py) / self.max_height
            dist_p = ((dx_p**2 + dy_p**2)**0.5)

        # Prepare state tensor for AI model
        type_feat = [0.0, 0.0, 0.0]
        type_feat[my_type] = 1.0
        state_tensor = torch.tensor(
            [dx_t, dy_t, dist_t, dx_p, dy_p, dist_p] + type_feat, dtype=torch.float
        )

        # Use AI model to decide action
        with torch.no_grad():
            q_values = self.ai_model(state_tensor)
        action = int(torch.argmax(q_values).item())

        # Update acceleration based on action
        if action == 0 and nearest_prey:  # Move toward prey
            tx, ty = nearest_prey
            self.acceleration_x = self.max_accel * (tx - px) / self.max_width
            self.acceleration_y = self.max_accel * (ty - py) / self.max_height
        elif action == 1 and nearest_threat:  # Move away from threat
            tx, ty = nearest_threat
            self.acceleration_x = self.max_accel * (px - tx) / self.max_width
            self.acceleration_y = self.max_accel * (py - ty) / self.max_height
        else:  # Random movement
            self.acceleration_x = random.uniform(-0.5, 0.5) * self.max_accel
            self.acceleration_y = random.uniform(-0.5, 0.5) * self.max_accel

    # Update position and handle movement logic
    def update_position(self, others=None):
        if self.ai_model and others is not None:
            self.decide_acceleration(others)
            self.ticks = 0
        else:
            self.ticks += 1
            if self.ticks == 5:
                self.time += 1
                self.ticks = 0
                target_x = random.randint(
                    trim((self.time * self.max_width // 100), self.max_width // 4, 0),
                    trim(self.max_width - (self.time * self.max_width // 100), self.max_width, 3 * self.max_width // 4))
                vector_x = target_x - self.coord_x
                self.acceleration_x = self.max_accel * vector_x / self.max_width

                target_y = random.randint(
                    trim((self.time * self.max_height // 100), self.max_height // 4, 0),
                    trim(self.max_height - (self.time * self.max_height // 100), self.max_height, 3 * self.max_height // 4))
                vector_y = target_y - self.coord_y
                self.acceleration_y = self.max_accel * vector_y / self.max_height

        # Update position, velocity, and acceleration
        self.coord_x += self.velocity_x
        self.coord_y += self.velocity_y
        self.velocity_x += self.acceleration_x
        self.velocity_y += self.acceleration_y

        # Constrain values within bounds
        self.coord_x = trim(self.coord_x, self.max_width, 0)
        self.coord_y = trim(self.coord_y, self.max_height, 0)
        self.velocity_x = trim(self.velocity_x, self.max_velo, -self.max_velo)
        self.velocity_y = trim(self.velocity_y, self.max_velo, -self.max_velo)
        self.acceleration_x = trim(self.acceleration_x, self.max_accel, -self.max_accel)
        self.acceleration_y = trim(self.acceleration_y, self.max_accel, -self.max_accel)

    def __str__(self):
        return f"{self.color} {self.coord_x} {self.coord_y}"