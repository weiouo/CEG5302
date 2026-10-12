import numpy as np

class ZDT3():

    def __init__(self):
        self.name = 'ZDT3'
        self.n_obj = 2  # number of objectives
        self.n_var = 30  # number of decision variables

        self.lower = np.zeros(self.n_var)  # lower bound of decision variables
        self.upper = np.ones(self.n_var)  # upper bound of decision variables

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj)) # for each chromosom, we have 2 objective function evals.
        f[:, 0] = x[:, 0]
        g = 1.0 + 9.0 * np.sum(x[:, 1:], axis=1) / (self.n_var - 1)
        h = 1 - np.power(
            f[:, 0] * 1.0 / g,
            0.5) - (f[:, 0] * 1.0 / g) * np.sin(10 * np.pi * f[:, 0])
        f[:, 1] = g * h

        return f

# Create an instance of ZDT3
prob = ZDT3()
print(f"Problem = {prob.name}")
print(f"Number of objectives = {prob.n_obj}")
print(f"Number of variables = {prob.n_var}")
print(f"Lower bounds = {prob.lower}")
print(f"Upper bounds = {prob.upper}")

# Initialize a population with 1 solution and evaluate it
pop_size = 1
x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
    pop_size, prob.n_var)

print(f"x: {x}")

f = prob.evaluate(x)
print(f"f(x) = {f}")

# Initialize a population with 5 solutions and evaluate them
pop_size = 5
x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
    pop_size, prob.n_var)


f = prob.evaluate(x)
print(f"f(x) = {f}")

class NSGA2():
    def __init__(self, pop_size=100):
        self.pop_size = pop_size
        #-----------BEGIN-----------#
        # TODO: Include other parameters you may use to create an 
        #       instance of NSGA2(), such as crossover probability 
        #       and mutation probability  
        #-----------END-----------#
        
    def initialize(self, prob):
        x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
            self.pop_size, prob.n_var)
        return x
    
    def fitness_assignment(self, population_fx):

        #-----------BEGIN-----------#
        # TODO: Implement non-dominated sorting and crowding distance
        # to assign non-domination rank and local crowding distance to each 
        # solution in the parent population, which will be used later in
        # tournament selection
        #-----------END-----------#

        # fast non-dominated sorting

        n = len(population_fx)
        rank = np.zeros(n, dtype=int)

        s_p = [[] for _ in range(n)] 
        n_p = np.zeros(n, dtype=int)
        fronts = [[]]

        for p in range(n):
            for q in range(n):

                if p==q:
                    continue #same individul

                if (np.all(population_fx[p] <= population_fx[q])and np.any(population_fx[p] < population_fx[q])):
                    s_p[p].append(q)

                elif (np.all(population_fx[q] <= population_fx[p])and np.any(population_fx[q] < population_fx[p])):
                    n_p[p]+=1

            if n_p[p] == 0:
                rank[p] = 1
                fronts[0].append(p)
        i = 0

        while len(fronts[i]) > 0:

            front_next = []

            for p in fronts[i]:
                for q in s_p[p]:

                    n_p[q]-=1
                    if n_p[q] == 0:
                        rank[q] = i+2
                        front_next.append(q)
            i += 1
            fronts.append(front_next)
        
        fronts.pop() # the last list is empty

        # crowding distance
        
        crowding_dist = np.zeros(n)

        for front in fronts:

            if len(front) <= 2:
                crowding_dist[front] = np.inf
                continue

            front = np.array(front)

            for m in range(population_fx.shape[1]):
                sorted_index = front[np.argsort(population_fx[front, m])]

                f_min = population_fx[sorted_index[0], m]
                f_max = population_fx[sorted_index[-1], m]

                if f_max == f_min:
                    continue

                crowding_dist[sorted_index[0]] = np.inf
                crowding_dist[sorted_index[-1]] = np.inf

                for i in range(1, len(sorted_index) - 1):

                    f_prev = population_fx[sorted_index[i - 1], m]
                    f_next = population_fx[sorted_index[i + 1], m]

                    crowding_dist[sorted_index[i]] += (f_next-f_prev)/(f_max-f_min)

        return rank, crowding_dist
    
    def tournament_selection(self, population_x, rank, crowding_dist):
        #-----------BEGIN-----------#
        # TODO: Implement tournament selection on parent population to  
        # construct a mating pool of size `pop_size` for crossover
        #-----------END-----------#
        mating_pool = np.zeros_like(population_x) #(pop_size,n_var) = (100,30)
        
        for i in range(self.pop_size):
            a, b = np.random.choice(len(population_x), 2, replace=False)
        
            if rank[a] < rank[b]:
                winner = a
        
            elif rank[b] < rank[a]:
                winner = b
        
            else:
                if crowding_dist[a] > crowding_dist[b]:
                    winner = a
        
                elif crowding_dist[b] > crowding_dist[a]:
                    winner = b
        
                else:
                    winner = np.random.choice([a,b])
        
            mating_pool[i] = population_x[winner]
        
        return mating_pool
    
    def crossover(self, parents, prob, pc):
        #-----------BEGIN-----------#
        # TODO: Implement simulated binary crossover (SBX) to generate 
        #       offspring of size `pop_size`
        #-----------END-----------#

        eta_c = 20 # decide the offspring how close to the parents
        offsprings  = parents.copy()

        for i in range(0,self.pop_size,2):
            parent1 = parents[i]
            parent2 = parents[i+1]

            if np.random.rand() <= pc:

                u = np.random.rand(prob.n_var) # random generate n_var variables

                #SBX
                beta = np.where(
                    u <= 0.5,
                    (2 * u) ** (1 / (eta_c + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                )

                offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

                offsprings[i] = np.clip(offspring1, prob.lower, prob.upper)
                offsprings[i + 1] = np.clip(offspring2, prob.lower, prob.upper)

        return offsprings
        
    def mutation(self, offsprings, prob, pm):
        #-----------BEGIN-----------#
        # TODO: Implement polynomial mutation (PM)
        #-----------END-----------#

        eta_m = 20

        for i in range(self.pop_size):
            for j in range(prob.n_var):

                if np.random.rand() <= pm:
                    x = offsprings[i, j]
                    lower = prob.lower[j]
                    upper = prob.upper[j]

                    delta1 = (x-lower) / (upper-lower)
                    delta2 = (upper-x) / (upper-lower)

                    u = np.random.rand()
                    mut_pow = 1.0 / (eta_m+1.0)

                    if u < 0.5:
                        xy = 1.0-delta1
                        val = 2.0 * u + (1.0-2.0*u) * (xy**(eta_m+1.0))
                        deltaq = val**mut_pow-1.0
                    else:
                        xy = 1.0-delta2
                        val = 2.0 * (1.0-u)+ 2.0 * (u-0.5) * (xy**(eta_m+1.0))
                        deltaq = 1.0-val**mut_pow

                    offsprings[i, j] = np.clip(x+deltaq*(upper-lower), lower, upper)

        return offsprings

        
    def environmental_selection(self, population_x, offspring_x, prob):
        #-----------BEGIN-----------#
        # TODO: Implement non-dominated sorting and crowding distance on
        # the combined population of parent population and offspring population
        # to select `pop_size` solutions for next population from combined population
        # of 2 * `pop_size` solutions
        #-----------END-----------#

        combined_x = np.vstack(((population_x, offspring_x))) # shape = (200,30)
        combined_fx = prob.evaluate(combined_x)
        rank, crowding_dist = self.fitness_assignment(combined_fx)
        sorted_index = np.lexsort((-crowding_dist, rank))

        next_generation = sorted_index[:self.pop_size]
        population_x = combined_x[next_generation]
        population_fx = combined_fx[next_generation]

        return population_x, population_fx
    
    def run(self, prob, max_gen=500):
        # Initialization
        population_x = self.initialize(prob)
        population_fx = prob.evaluate(population_x)
                
        # Evolve until termination condition is met

        for i in range(1, max_gen):
            #-----------BEGIN-----------#
            # TODO:
            # fitness assignment of parent population
            rank, crowding_dist = self.fitness_assignment(population_fx)

            # parent = self.tournament_selection(?)
            parents = self.tournament_selection(population_x, rank, crowding_dist)

            # offspring = self.crossover(??)
            offspring_x = self.crossover(parents, prob, pc=0.8)

            # offspring_x = self.mutation(???)
            offspring_x = self.mutation(offspring_x, prob, pm=0.05)

            # offspring_fx = prob.evaluate(offspring_x)
            

            # Apply enviromental selection to select solutions
            [population_x, population_fx]  = self.environmental_selection(population_x, offspring_x,prob)
       
        return population_x, population_fx

# Instantiate class NSGA2() and class ZDT3()
optimizer = NSGA2(pop_size=100)
problem = ZDT3()

# Use NSGA-II to solve the ZDT3 problem
[optimum_x, optimum_fx] = optimizer.run(problem, max_gen=500)

#-----------BEGIN-----------#
# TODO: Plot the final population `optimum_fx` in the objective space
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    optimum_fx[:, 0],
    optimum_fx[:, 1],
    s=15,
    label="NSGA-II"
)

plt.xlabel("f1")
plt.ylabel("f2")
plt.title("NSGA-II on ZDT3")
plt.legend()
plt.grid(True)

plt.show()
#-----------END-----------#

'''
class CrashworthinessDesign():

    def __init__(self):
        self.name = 'Crashworthiness design of vehicles'
        self.n_obj = 3
        self.n_var = 5

        self.lower = np.full(self.n_var, 1.0)
        self.upper = np.full(self.n_var, 3.0)

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj))

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        f[:, 0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (
            4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f[:, 1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (
            0.8364 * x4) - (0.3695 * x1 * x4) + (0.0861 * x1 * x5) + (
                0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                    0.1764 * x4 * x4)
        f[:, 2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (
            0.0073 * x1 * x2) + (0.024 * x2 * x3) - (0.0118 * x2 * x4) - (
                0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (
                    0.0109 * x4 * x4)

        return f

# Create an instance of the crashworthiness design problem
prob = CrashworthinessDesign()
print(f"Problem = {prob.name}")
print(f"Number of objectives = {prob.n_obj}")
print(f"Number of variables = {prob.n_var}")
print(f"Lower bounds = {prob.lower}")
print(f"Upper bounds = {prob.upper}")

# Initialize a population with 5 solutions and evaluate them
pop_size = 5
x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
    pop_size, prob.n_var)
print("x = ", x)

f = prob.evaluate(x)
print(f"f(x) = {f}")

#-----------BEGIN-----------#
# TODO: 
#   1. Instantiate class NSGA2() and class CrashworthinessDesign()
#   2. Use NSGA-II to solve the crashworthiness design problem
#   3. Plot the final population `optimum_fx` in the objective space
#   4. Share your thoughts on the questions above.
#-----------END-----------#

class MW7():

    def __init__(self):
        self.name = 'MW7'
        self.n_obj = 2  # number of objectives
        self.n_var = 15  # number of decision variables
        self.n_con = 2  # number of constraints

        self.lower = np.zeros(self.n_var)
        self.upper = np.ones(self.n_var)

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj))
        c = np.zeros((pop_size, self.n_con))

        g3 = 1 + 2.0 * np.sum((x[:, self.n_obj - 1:] +
                               (x[:, self.n_obj - 2:-1] - 0.5)**2 - 1.0)**2,
                              axis=1)

        f[:, 0] = g3 * x[:, 0]
        f[:, 1] = g3 * (1 - (f[:, 0] / g3)**2)**0.5

        with np.errstate(divide='ignore'):
            l = np.arctan(f[:, 1] / f[:, 0])

        c[:, 0] = (1.2 + 0.4 * (np.sin(4 * l))**16)**2 - f[:, 0]**2 - f[:, 1]**2
        c[:, 1] = f[:, 0]**2 + f[:, 1]**2 - (1.15 - 0.2 * (np.sin(4 * l))**8)**2
        c = np.where(c < 0, -c, 0)  # the degree of constraint violation of x at each constraint

        return f, c

# Create an instance of MW7
prob = MW7()
print(f"Problem = {prob.name}")
print(f"Number of objectives = {prob.n_obj}")
print(f"Number of variables = {prob.n_var}")
print(f"Number of constraints = {prob.n_con}")
print(f"Lower bounds = {prob.lower}")
print(f"Upper bounds = {prob.upper}")

# Initialize a population with 1 solution and evaluate it
pop_size = 1
x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
    pop_size, prob.n_var)
print("x = ", x)

f, c = prob.evaluate(x)
print(f"f(x) = {f}")
print(f"c(x) = {c}")
print(f"CV(x) = {np.sum(c, axis=1, keepdims=True)}")  # Constraint Violation

# Initialize a population with 2 solution and evaluate them
pop_size = 2
x = prob.lower + (prob.upper - prob.lower) * np.random.rand(
    pop_size, prob.n_var)
print("x = ", x)

f, c = prob.evaluate(x)
print(f"f(x) = {f}")
print(f"c(x) = {c}")
print(f"CV(x) = {np.sum(c, axis=1, keepdims=True)}")

class CNSGA2():

'''
    #-----------BEGIN-----------#
    # TODO: Implement the parameter-less constraint handling technique in NSGA-II
    #-----------END-----------#

#-----------BEGIN-----------#
# TODO: 
#   1. Instantiate class CNSGA2() and class MW7()
#   2. Use NSGA-II with a contraint handling technique to solve the MW7 problem
#   3. Plot the final population `optimum_fx` in the objective space
#   4. In your report, answer the questions like: What are the possible reasons why the solutions 
#      obtained by NSGA-II fail to cover all segments of the Pareto front?
#-----------END-----------#

#-----------BEGIN-----------#
# TODO:
#   1. Implement the problem class for the assigned RCMx real-world problem
#   2. Instantiate the CNSGA2() class and the problem class
#   3. Use the constrained NSGA-II algorithm to solve the problem with a range of parameter settings
#   4. Analyze the performance of the algorithm with different parameter settings, including the 
#      population size, crossover and mutation probabilities, and maximum number of generations
#   5. Plot the final population `optimum_fx` in the objective space  using the best 
#      parameter settings that you found in Step 4.
#-----------END-----------#