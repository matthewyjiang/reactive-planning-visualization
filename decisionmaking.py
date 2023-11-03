import numpy as np

class pcm_model():
    
    max_value = 1000
    
    def __init__(self):
        self.undecided_m = 1
        self.tasks = np.array([])
        self.sigma_array = np.array([])
        self.v_star_array = np.array([])
        self.grads = []
        self.motivations = np.array([])
        self.values = np.array([])
        self.output_scale = 1
        self.epsilon = 0.0001
        
        
    def add_task(self, task):
        self.tasks = np.append(self.tasks,task)
        self.grads.append(task.get_grad)
        self.motivations = np.append(self.motivations, 0)
        self.values = np.append(self.values, 0.001)
        self.sigma_array = np.append(self.sigma_array, task.sigma)
        self.v_star_array = np.append(self.v_star_array, task.v_star)

        
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
        v = np.array(self.v_star_array) * np.array(self.values)
        v_with_epsilon = np.maximum(v, epsilon)
        part1 = v * self.undecided_m
        part2 = (1 / v_with_epsilon)
        part3 = v * self.undecided_m
        part4 = self.sigma_array * (1 - self.motivations - self.undecided_m)
        motivation_change = part1 - self.motivations*(part2 - part3 + part4)

        # Preserve the total motivation
        total_motivation = self.motivations.sum() + self.undecided_m

        # Clip the motivation change to ensure values are between 0 and 1
        motivation_change_clipped = np.clip(motivation_change, -self.motivations, 1 - self.motivations)

        # Update motivations with the clipped changes
        self.motivations += 0.1*motivation_change_clipped

        # Adjust undecided_m to account for the clipped changes in motivations to preserve total motivation
        self.undecided_m = total_motivation - self.motivations.sum()
            
            
    
    def update_values(self, x):
        print(self.values)
        for i in range(len(self.values)):
            value_change = self.tasks[i].lambd*(self.tasks[i].get_phi(x)-self.values[i])
            
            self.values[i] += value_change
            if(self.values[i] > self.max_value):
                self.values[i] = self.max_value
            if(self.values[i] < 0):
                self.values[i] = 0
        