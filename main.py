from task import task
from decisionmaking import pcm_model
import numpy as np
import matplotlib.pyplot as plt

tasks = []

def phi(x, x0):
    return np.linalg.norm(x-x0)

def grad(x, x0):
    return x-x0

if __name__ == "__main__":
    
    # create 5 tasks with random coordinates and equal value gain
    d = 2
    
    for i in range(5):
        x = np.random.randint(0, 50, d)
        tasks.append(task(x, np.random.randint(0, 10), phi, grad))
    
    model = pcm_model()
    
    for t in tasks:
        model.add_task(t)
        
    while(True):
        model.update_values()
        model.update_motivation()
        
    
        
    