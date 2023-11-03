from entity import Entity
import math
import pygame
import numpy as np

class Robot(Entity):
    def __init__(self, x, y, mass, friction):
        super().__init__(x, y, 20, 20, mass)
        self.friction = friction
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = 0
        self.theta = 0
        self.max_speed = 10

    def apply_force(self, fx, fy):
        self.ax += fx / self.mass
        self.ay += fy / self.mass

    def add_force_in_theta_direction(self, f):
        fx = f * math.cos(self.theta)
        fy = f * math.sin(self.theta)
        self.apply_force(fx, fy)
    
    def set_speed(self, vx, vy):
        self.vx = vx
        self.vy = vy
            
        
    def get_coordinates(self):
        return np.array((self.x + 10, self.y + 10))
    
    def update(self, dt):
        super().update(dt)
        self.vx -= self.vx * self.friction * dt
        self.vy -= self.vy * self.friction * dt
        if self.vx > self.max_speed:
            self.vx = self.max_speed
            
        if self.vy > self.max_speed:
            self.vy = self.max_speed

    def draw(self, screen):
        # Draw a rectangle
        rect = pygame.Rect(self.x, self.y, self.w, self.h)
        pygame.draw.rect(screen, (255, 0, 0), rect)
