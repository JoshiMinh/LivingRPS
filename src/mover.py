import random

def trim(value, upper, lower):
    """Clamp value between lower and upper bounds."""
    return max(lower, min(value, upper))

class Mover:
    """
    Represents an agent in a Rock-Paper-Scissors simulation with position, velocity, and acceleration.
    Handles movement, interaction, and state representation.
    """
    def __init__(self, thing, coords, v, a, max_height, max_width, max_velo, max_accel):
        self.ticks = 0
        self.time = 0
        self.color = thing  # 0: Rock, 1: Scissors, 2: Paper
        self.coord_x, self.coord_y = coords
        self.velocity_x, self.velocity_y = v
        self.acceleration_x, self.acceleration_y = a
        self.max_height = max_height
        self.max_width = max_width
        self.max_velo = max_velo
        self.max_accel = max_accel

    def get_color(self):
        """Return the current type (color) of the mover."""
        return self.color

    def update_color(self, c):
        """
        Update the mover's color if the incoming type beats the current type.
        Winning rules: 0 (Rock) beats 1 (Scissors), 1 beats 2 (Paper), 2 beats 0 (Rock).
        """
        if (c, self.color) in [(0, 1), (1, 2), (2, 0)]:
            self.color = c

    def get_position(self):
        """Return the current (x, y) position."""
        return self.coord_x, self.coord_y

    def get_state(self, others):
        """
        Return a feature vector representing the mover's state:
        - Relative position and distance to nearest threat and prey
        - One-hot encoding of own type
        """
        my_type = self.color
        # Determine threat and prey types based on current type
        threat_type, prey_type = (
            (2, 1) if my_type == 0 else
            (0, 2) if my_type == 1 else
            (1, 0)
        )

        nearest_threat, nearest_prey = None, None
        min_d_threat, min_d_prey = float('inf'), float('inf')
        px, py = self.coord_x, self.coord_y

        for other in others:
            if other is self:
                continue
            ox, oy = other.get_position()
            d2 = (ox - px) ** 2 + (oy - py) ** 2
            if other.get_color() == threat_type and d2 < min_d_threat:
                min_d_threat, nearest_threat = d2, (ox, oy)
            if other.get_color() == prey_type and d2 < min_d_prey:
                min_d_prey, nearest_prey = d2, (ox, oy)

        # Default values if no threat/prey found
        dx_t, dy_t, dist_t = 0.0, 0.0, 1.0
        dx_p, dy_p, dist_p = 0.0, 0.0, 1.0

        if nearest_threat:
            dx_t = (nearest_threat[0] - px) / self.max_width
            dy_t = (nearest_threat[1] - py) / self.max_height
            dist_t = (dx_t ** 2 + dy_t ** 2) ** 0.5
        if nearest_prey:
            dx_p = (nearest_prey[0] - px) / self.max_width
            dy_p = (nearest_prey[1] - py) / self.max_height
            dist_p = (dx_p ** 2 + dy_p ** 2) ** 0.5

        # One-hot encoding for type
        type_feat = [0.0, 0.0, 0.0]
        type_feat[my_type] = 1.0

        return [dx_t, dy_t, dist_t, dx_p, dy_p, dist_p] + type_feat

    def apply_action(self, action, others):
        """
        Apply an action to the mover:
        - action == 0: Move toward nearest prey
        - action == 1: Move away from nearest threat
        - otherwise: Move randomly
        """
        px, py = self.coord_x, self.coord_y
        my_type = self.color
        threat_type, prey_type = (
            (2, 1) if my_type == 0 else
            (0, 2) if my_type == 1 else
            (1, 0)
        )

        tx, ty = None, None

        if action == 0:  # Move toward prey
            min_d, target = float('inf'), None
            for other in others:
                if other.get_color() == prey_type:
                    ox, oy = other.get_position()
                    d2 = (ox - px) ** 2 + (oy - py) ** 2
                    if d2 < min_d:
                        min_d, target = d2, (ox, oy)
            if target:
                tx, ty = target

        elif action == 1:  # Move away from threat
            min_d, target = float('inf'), None
            for other in others:
                if other.get_color() == threat_type:
                    ox, oy = other.get_position()
                    d2 = (ox - px) ** 2 + (oy - py) ** 2
                    if d2 < min_d:
                        min_d, target = d2, (ox, oy)
            if target:
                # Move in the opposite direction from the threat
                tx, ty = px - (target[0] - px), py - (target[1] - py)

        if tx is not None:
            self.acceleration_x = self.max_accel * (tx - px) / self.max_width
            self.acceleration_y = self.max_accel * (ty - py) / self.max_height
        else:
            # Random movement if no target found
            self.acceleration_x = random.uniform(-0.5, 0.5) * self.max_accel
            self.acceleration_y = random.uniform(-0.5, 0.5) * self.max_accel

    def update_physics(self):
        """
        Update the mover's position, velocity, and acceleration.
        Clamp values to their respective maximums and boundaries.
        """
        self.coord_x += self.velocity_x
        self.coord_y += self.velocity_y
        self.velocity_x += self.acceleration_x
        self.velocity_y += self.acceleration_y

        self.coord_x = trim(self.coord_x, self.max_width - 30, 0)
        self.coord_y = trim(self.coord_y, self.max_height - 30, 0)
        self.velocity_x = trim(self.velocity_x, self.max_velo, -self.max_velo)
        self.velocity_y = trim(self.velocity_y, self.max_velo, -self.max_velo)
        self.acceleration_x = trim(self.acceleration_x, self.max_accel, -self.max_accel)
        self.acceleration_y = trim(self.acceleration_y, self.max_accel, -self.max_accel)

    def __str__(self):
        """String representation: type and position."""
        return f"{self.color} {self.coord_x} {self.coord_y}"