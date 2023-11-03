import numpy as np

class pcm_model():
    
    max_value = 10000
    
    def __init__(self):
        self.undecided_m = 1
        self.tasks = []
        self.grads = []
        self.motivations = []
        self.values = []
        self.output_scale = 1
        self.epsilon = 0.0001
        
    def add_task(self, task):
        self.tasks.append(task)
        self.grads.append(task.get_grad)
        self.motivations.append(0)
        self.values.append(0.001)
        
    def get_navigation_output(self, x):
        grads = np.array([g(x) for g in self.grads])
        motivations = np.array(self.motivations)
        return -self.output_scale*np.dot(grads.T, motivations)
    
    def normalize_motivations(self):
        """
        Normalize the motivations so they sum up to 1.
        """
        total = np.sum(self.motivations) + self.undecided_m
        if total != 0:
            self.motivations /= total
        else:
            # Handle the case when sum is 0.
            self.motivations = np.zeros_like(self.m)
            self.undecided_m = 1.0  # Set undecided motivation to 1
    
    def update_motivation(self):
        epsilon = 1e-10  # A small number to prevent division by zero
        for i in range(len(self.motivations)):
            v = self.tasks[i].v_star*self.values[0]
            epsilon = 1e-10
            v_with_epsilon = np.maximum(v, epsilon)

            # Decompose the expression into smaller parts to identify where the invalid value might be generated
            part1 = v * self.undecided_m
            part2 = self.motivations[i] * (1 / v_with_epsilon)
            part3 = self.motivations[i] * v * self.undecided_m
            part4 = self.motivations[i] * self.tasks[i].sigma * (1 - self.motivations[i] - self.undecided_m)

            # Now assemble the parts with careful checks
            motivation_change = part1 - (part2 - part3 + part4)
                
            if self.motivations[i] + motivation_change < 0:
                self.undecided_m += self.motivations[i]
                self.motivations[i] = 0
            elif self.motivations[i] + motivation_change > 1:
                self.undecided_m += 1 - self.motivations[i]
                self.motivations[i] = 1
            elif self.undecided_m - motivation_change < 0:
                self.motivations[i] += self.undecided_m
                self.undecided_m = 0
            elif self.undecided_m - motivation_change > 1:
                self.motivations[i] += 1 - self.undecided_m
                self.undecided_m = 1
            else:
                self.motivations[i] += motivation_change
                self.undecided_m -= motivation_change 

            self.normalize_motivations()
            #normalize
            
            
    
    def update_values(self, x):
        print(self.values)
        for i in range(len(self.values)):
            value_change = self.tasks[i].lambd*(self.tasks[i].get_phi(x)-self.values[i])
            
            self.values[i] += value_change
            if(self.values[i] > self.max_value):
                self.values[i] = self.max_value
            if(self.values[i] < 0):
                self.values[i] = 0
        