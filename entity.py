class Entity:
    def __init__(self, x, y, w, h, mass):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mass = mass
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0

    def apply_force(self, fx, fy):
        self.ax += fx / self.mass
        self.ay += fy / self.mass

    def update(self, dt):
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.ax = 0
        self.ay = 0

    def draw(self):
        # TODO: Implement draw function
        pass
