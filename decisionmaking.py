import numpy as np

class pcm_model():
    
    def __init__(self):
        self.undecided_m = 10
        self.tasks = []
        self.grads = []
        self.motivations = []
        self.values = []
        
    def add_task(self, task):
        self.tasks.append(task)
        self.grads.append(task.get_grad)
        self.motivations.append(0)
        self.values.append(0)
        
    def get_navigation_output(self, x):
        grads = np.array([g(x) for g in self.grads])
        motivations = np.array(self.motivations)
        return -np.dot(grads, motivations)
    
    def update_motivation(self):
        for i in range(len(self.motivations)):
            v = self.tasks[i].v_star*self.values[0]
            self.motivations[i] = v*self.undecided_m - self.motivations[i](1/v-v*self.undecided_m+self.tasks[i].sigma(1-self.motivations[i]-self.undecided_m))
    
    def update_values(self):
        for i in range(len(self.values)):
            self.values[i] = self.tasks[i].lambd*(self.tasks[i].get_phi(self.x)-self.values[i])
        