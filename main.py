import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler

from model import Model
from load import LAC_feature


def Load_Model(num_feat):
    device = torch.device("cuda")
    model = Model(num_feat)
    model = model.to(device)
    model.load_state_dict(torch.load("model.pt"))
    return model


def Initialization(population_size):
    population = []
    for _ in range(population_size):
        individual = []
        for _ in range(9):
            time = random.uniform(0.0, 48.0)
            individual.append(time)
        population.append(individual)
    return population


def Tournament_Selection(population):
    tornament_size = 2
    parents = []
    for _ in range(tornament_size):
        indice = np.random.choice(a=len(population), size=2)
        individual1 = population[indice[0]]
        individual2 = population[indice[1]]
        if Fitness(individual1) > Fitness(individual2):
            parents.append(individual1)
        else:
            parents.append(individual2)

    return parents


def Crossover(parents):
    # """One-point crossover with probability pc = 1.0"""
    # cross_point = random.randint(1, 8)
    # child0 = parents[0][:cross_point] + parents[1][cross_point:]
    # child1 = parents[1][:cross_point] + parents[0][cross_point:]
    # return [child0, child1]

    """Uniform crossover"""
    child0 = []
    child1 = []
    for i in range(len(parents[0])):
        idx = random.randint(0, 1)
        if idx == 0:
            child0.append(parents[0][i])
            child1.append(parents[1][i])
        else:
            child0.append(parents[1][i])
            child1.append(parents[0][i])

    return [child0, child1]


def Mutation(individual):
    """Gaussian Mutation"""
    z = np.random.normal(0, 0.01, len(individual))
    new_indi = individual + z

    if Fitness(new_indi) > Fitness(individual):
        return new_indi
    else:
        return individual


def Fitness(individual):
    global num_feat, player_states
    """Penalize individual that contains the sum which is fewer than 240 minutes in total"""
    penalty = abs(240.0 - sum(individual)) / 240.0
    penalty = round(penalty, 5)

    """Modified players' weight to create a sense of fatigue"""
    modified_individual = individual.copy()
    for i in range(len(individual)):
        if individual[i] > avg_time[i]:
            time_exceed = individual[i] - avg_time[i]
            # Player can only have a portion of avg performance when playing more than avg time
            modified_individual[i] = avg_time[i] + 0.3 * time_exceed

    """Get LAC's weighted states"""
    weighted_states = np.zeros((1, num_feat))
    # ith feature
    for i in range(num_feat):
        # jth player
        for j in range(len(player_states)):
            weighted_states[0][i] += (modified_individual[j] / 48.0) * player_states[j][
                i
            ]

    # weighted_states = StandardScaler().fit_transform(weighted_states.T).T
    weighted_states = weighted_states.astype(np.float32)

    device = torch.device("cuda")
    weighted_states = torch.tensor(weighted_states)
    weighted_states = weighted_states.to(device)

    model.eval()
    with torch.no_grad():
        predict_winrate = model(weighted_states)
    predict_winrate = round(predict_winrate.item(), 5)

    fitness_value = predict_winrate - penalty

    global best_individual, best_fitness_value, best_winrate, best_penalty
    if fitness_value > best_fitness_value:
        best_individual = individual
        best_fitness_value = fitness_value
        best_winrate = predict_winrate
        best_penalty = penalty

        # print(f"Better Individual: {best_winrate}, {best_penalty}")

    return fitness_value, predict_winrate, penalty


runs = 10
population_size = 100
generations = 100
player_states, avg_time, num_feat = LAC_feature()
model = Load_Model(num_feat)


best_individual = None
best_fitness_value = None
best_winrate = None
best_penalty = None

if __name__ == "__main__":
    avg_individual = [0.0] * 9
    avg_fitness = np.zeros(generations)
    avg_winrate = np.zeros(generations)
    avg_penalty = np.zeros(generations)
    for _ in range(runs):
        best_individual = []
        best_fitness_value = -1 * np.inf
        best_winrate = 0.0
        best_penalty = 0.0

        generation_fitness = np.zeros(generations)
        generation_winrate = np.zeros(generations)
        generation_penalty = np.zeros(generations)
        population = Initialization(population_size)

        for generation in range(generations):
            value = [Fitness(individual) for individual in population]
            fitness = [a[0] for a in value]
            winrate = [a[1] for a in value]
            penalty = [a[2] for a in value]

            best_idx = fitness.index(max(fitness))
            best_fitness = fitness[best_idx]
            generation_fitness[generation] = best_fitness

            best_winrate = winrate[best_idx]
            generation_winrate[generation] = best_winrate

            best_penalty = penalty[best_idx]
            generation_penalty[generation] = best_penalty

            next_population = []
            for i in range(len(population) // 2):
                parents = Tournament_Selection(population)
                children = Crossover(parents)
                children = [Mutation(child) for child in children]
                next_population.extend(children)

            population = next_population

        avg_fitness += generation_fitness / runs
        avg_winrate += generation_winrate / runs
        avg_penalty += generation_penalty / runs
        avg_individual = [
            round(x + y, 5) for x, y in zip(avg_individual, best_individual)
        ]

        print(f"Best Individual: {best_individual}")
        print(f"Best Fitness: {best_fitness_value}")
        print(f"Best WinRate: {best_winrate}")
        print(f"Best Penalty: {best_penalty}")

    avg_individual = [x / runs for x in avg_individual]
    print(f"Average Best Time: {avg_individual}")

    plt.plot(range(generations), avg_fitness, label="Fitness", color="blue")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Average of the best fitness values")
    plt.title("WinRate Prediction")
    plt.show()
    plt.clf()

    plt.plot(range(generations), avg_winrate, label="WinRate", color="green")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Average of the best winrate")
    plt.title("WinRate Prediction")
    plt.show()
    plt.clf()

    plt.plot(range(generations), avg_penalty, label="Penalty", color="red")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Average of the lowest penalty")
    plt.title("WinRate Prediction")
    plt.show()
    plt.clf()
