import re
import random
import time
import math
with open("a280.tsp", "r") as file:
    contents = file.read()
    print(contents)

def main():
    # read in the user input for the algorithm and time limit
    algorithm = input("Please select an algorithm (SA, TS, or GA): ")
    time_limit = int(input("Please enter a time limit (between 60 and 300 seconds): "))

    with open("a280.tsp", "r") as file:
        contents = file.read()
    

        
    # parse the TSP instance file and get the list of coordinates
    coords = parse_tsp_file(contents)
    
    # create the distance matrix
    distance_matrix = calculate_distance_matrix(coords)
    
    # obtain the initial solution
    initial_solution = nearest_neighbor(coords, distance_matrix)

    if algorithm == 'SA':
        best_solution = simulated_annealing(distance_matrix, initial_solution, time_limit)
    elif algorithm == 'TS':
        best_solution = tabu_search(distance_matrix, initial_solution, time_limit)
    elif algorithm == 'GA':
        best_solution = genetic_algorithm(distance_matrix, 50, 0.1, time_limit)
    
    print(best_solution)
    print(calculate_cost(distance_matrix, best_solution))

def calculate_cost(distance_matrix, solution):
    cost = 0
    for i in range(len(solution) - 1):
        
        cost += distance_matrix[solution[i]][solution[i + 1]]
    return cost

# function to parse the contents of a TSP instance file and return a list of node coordinates
def parse_tsp_file(contents):
    # split the contents into lines
    lines = contents.split("\n")
    
    # empty list to store node coordinates
    coords = []
    
    # flag to indicate whether we have reached the NODE_COORD_SECTION
    found_coords = False
    
    # loop through the lines
    for line in lines:
        # if we have reached the NODE_COORD_SECTION
        if found_coords:
            # use a regular expression to extract the node number and coordinates
            match = re.search(r"(\d+)\s+([-\d.]+)\s+([-\d.]+)", line)
            # if a match is found
            if match:
                # add the coordinates to the list
                coords.append((float(match.group(2)), float(match.group(3))))
        # if we have reached the end of the NODE_COORD_SECTION
        elif line == "EOF":
            break
        # if we have found the NODE_COORD_SECTION
        elif line == "NODE_COORD_SECTION":
            found_coords = True
    
    return coords

# function to calculate the euclidean distance between two nodes
def euclidean_distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# function to calculate the distance matrix for a given list of node coordinates
def calculate_distance_matrix(coords):
    n = len(coords)
    distance_matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix[i][j] = float("inf")
            else:
                distance_matrix[i][j] = distance_matrix[j][i] = euclidean_distance(coords[i], coords[j])
    return distance_matrix

def nearest_neighbor(coords, distance_matrix):
    # start from a random node
    current_node = random.randint(0, len(coords) - 1)
    tour = [current_node]
    unvisited_nodes = set(range(len(coords)))
    unvisited_nodes.remove(current_node)
    
    while unvisited_nodes:
        nearest_node = None
        nearest_distance = float("inf")
        for node in unvisited_nodes:
            if distance_matrix[current_node][node] < nearest_distance:
                nearest_node = node
                nearest_distance = distance_matrix[current_node][node]
        current_node = nearest_node
        tour.append(current_node)
        unvisited_nodes.remove(current_node)
    
    return tour   

def random_solution(coords):
    # create a list of node indices
    nodes = list(range(len(coords)))
    # shuffle the list of nodes
    random.shuffle(nodes)
    # return the shuffled list of nodes
    return nodes
def acceptance_probability(current_cost, new_cost, temperature):
    return math.exp(-(new_cost - current_cost) / temperature)

def simulated_annealing(distance_matrix, initial_solution, time_limit):
    current_solution = initial_solution
    current_cost = calculate_cost(distance_matrix, current_solution)
    best_solution = current_solution
    best_cost = current_cost

    T0 = 1
    alpha = 0.995
    t = 1
    
    start_time = time.time()
    while True:
        if time.time() - start_time > time_limit:
            break
        # Generate new solution and calculate the cost
        new_solution = nearest_neighbor(current_solution, distance_matrix)
        new_cost = calculate_cost(distance_matrix, new_solution)
        if acceptance_probability(current_cost, new_cost, t) > random.random():
            current_solution = new_solution
            current_cost = new_cost
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
        t *= alpha
        
    return best_solution
def tabu_search(distance_matrix, initial_solution, time_limit):
    current_solution = initial_solution
    current_cost = calculate_cost(distance_matrix, current_solution)
    best_solution = current_solution
    best_cost = current_cost
    tabu_list = []
    tabu_list_size = len(initial_solution) // 2
    start_time = time.time()
    while time.time() - start_time < time_limit:
        best_neighbor = None
        best_neighbor_cost = float("inf")
        for i in range(len(initial_solution)-1):
            neighbor=swap_nodes(current_solution,i)
            if neighbor in tabu_list:
                continue
            cost = calculate_cost(distance_matrix, neighbor)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost
        
        current_solution = best_neighbor
        current_cost = best_neighbor_cost
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
    return best_solution

def swap_nodes(solution,i):
    new_sol=solution.copy()
    temp=new_sol[i]
    new_sol[i]=new_sol[i+1]
    new_sol[i+1]=temp
    return new_sol

def genetic_algorithm(distance_matrix, population_size, mutation_rate, time_limit):
    population = [random_solution(distance_matrix) for _ in range(population_size)]
    best_solution = None
    best_cost = float("inf")
    
    start_time = time.time()
    while time.time() - start_time < time_limit:
        # evaluate the fitness of each individual in the population
        population = sorted(population, key=lambda individual: calculate_cost(distance_matrix, individual))
        # select the top individuals for breeding
        breeding_pool = population[:population_size // 2]
        # generate the next generation of individuals
        next_generation = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(breeding_pool, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)
        population = next_generation
        
        # update the best solution
        if calculate_cost(distance_matrix, population[0]) < best_cost:
            best_solution = population[0]
            best_cost = calculate_cost(distance_matrix, population[0])

    return best_solution

def crossover(parent1, parent2):
    child = [-1 for _ in parent1]
    start, end = random.sample(range(len(parent1)), 2)
    start, end = min(start, end), max(start, end)
    child[start:end] = parent1[start:end]
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child
def mutate(solution):
    # select two random positions in the tour
    i, j = random.sample(range(len(solution)), 2)
    # swap the nodes at these positions
    solution[i], solution[j] = solution[j], solution[i]
    return solution



if __name__ == "__main__":
    main()