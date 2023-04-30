import pygad.torchga as torchga
import pygad
from GAmodel import GAmodel, device
import gym
import torch

env = gym.make("LunarLander-v2", render_mode=None)
model = GAmodel().to(device)
torch_ga = torchga.TorchGA(model = model, num_solutions=100)

def fitness_func(solution, sol_idx):
    global model, evn, torch_ga
    model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(model_weights_dict)

    observation, info = env.reset(seed=42)
    reward_sum = 0
    while True:
        action = model.get_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        observation = new_observation
        if terminated or truncated:
            break
    return reward_sum + 1000

def callback_generation(ga_instance):
    global env, model
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]-1000))
    best_solution = ga_instance.best_solution()[0]
    best_solution = torchga.model_weights_as_dict(model=model, weights_vector=best_solution)

    if ga_instance.best_solution()[1]-1000 > 200:
        model.load_state_dict(best_solution)
        env_show = gym.make("LunarLander-v2", render_mode="human")
        observation, info = env.reset(seed=42)
        reward_sum = 0
        while True:
            action = model.get_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            observation = new_observation
            if terminated or truncated:
                break
        print(f"nice solution reward: {reward_sum}")
        env_show.close()
        torch.save(model.state_dict(), "./checkpoint")
        ga_instance.plot_result(title="Iteration vs. Fitness")
        exit()

def callbacll_fitness(ga_instance, solusion_fits):
    pass
    #min_fits = min(ga_instance.last_generation_fitness)
    #ga_instance.last_generation_fitness = ga_instance.last_generation_fitness - min_fits +1 if min_fits<0 else ga_instance.last_generation_fitness

num_generations = 100
num_parents_mating = 5
initial_population = torch_ga.population_weights
parent_selection_type = "sss"
crossover_type = "single_point"
crossover_probability = 0.7
mutation_type = "random"
mutation_probability = 0.25
keep_parents = 1

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       keep_parents=keep_parents,
                       on_generation=callback_generation,
                       on_fitness=callbacll_fitness)
ga_instance.run()
env.close()