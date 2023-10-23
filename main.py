from task import task
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
    
        
    # plot the tasks
    
    # 50 x 50 linspace matrix
    
    x, y = np.meshgrid(
    np.arange(0, 50, 50),
    np.arange(0, 50, 50),
    sparse=False)
    
    
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Tasks')
    ax.grid(True)
    
    points = [[x, y] for x in range(0, 50, 1) for y in range(0, 50, 1)]
    vals = [tasks[0].get_grad(p) for p in points]
    ax.quiver([p[0] for p in points], [p[1] for p in points], [v[0] for v in vals], [v[1] for v in vals])
    ax.plot([t.x[0] for t in tasks], [t.x[1] for t in tasks], 'ro')
      
    plt.show()
        
    