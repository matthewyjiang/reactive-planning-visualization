import pygame
import robot
import numpy as  np
from decisionmaking import pcm_model
from task import task
import time

entities = []
texts = []

tasks = []

def phi(x, x0):
    return np.linalg.norm(x0-x)

def grad(x, x0):
    return x0-x

tasks.append(task(np.array([600, 200]), 0.01, phi, grad))
tasks.append(task(np.array([200, 400]), 0.01, phi, grad))

model = pcm_model()

for t in tasks:
    model.add_task(t)




my_robot = robot.Robot(x=250, y=250, mass=10, friction=0.5)

error = 0
integral = 0
target = 750
last_error = 0

def update_game():
    dt=1/60
    model.update_values(my_robot.get_coordinates())
    model.update_motivation()
    my_robot.set_speed(*model.get_navigation_output(my_robot.get_coordinates()))
    # time.sleep()
    
    
    
    # Update all entities
    for entity in entities:
        entity.update(dt=1/60)


def main():
    # Initialize pygame
    pygame.init()

    entities.append(my_robot)

    # Set up the display
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Robo Sim 2D")

    # print robot coordinates using draw text
    font = pygame.font.SysFont(None, 36)

    # Run the game loop
    clock = pygame.time.Clock()
    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Update the game
        update_game()

        # Draw the entities
        screen.fill((255, 255, 255))
        for entity in entities:
            entity.draw(screen)

        for task in tasks:
            pygame.draw.circle(screen, (0, 0, 0), task.x, 5, 2)
        
        # Get robot coordinates
        robot_x = my_robot.get_coordinates()[0]
        # draw force vector
        pygame.draw.line(screen, (0, 0, 0), my_robot.get_coordinates(), my_robot.get_coordinates()+model.get_navigation_output(my_robot.get_coordinates())*10)
        texts.clear()

        # Render text
        text = font.render("values X:" + str(np.round(model.values,2)),  1, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = 100
        textpos.centery = 20
        
        # Render text
        text1 = font.render("motivations:" + str(np.round(model.motivations,2)),  1, (10, 10, 10))
        textpos1 = text1.get_rect()
        textpos1.centerx = 100
        textpos1.centery = 60
        
        texts.append((text, textpos))
        texts.append((text1, textpos1))


        
        
        
        for text, textpos in texts:
            screen.blit(text, textpos)

        # Update the display
        pygame.display.flip()

        # Limit the frame rate
        clock.tick(60)


if __name__ == "__main__":
    main()
    

