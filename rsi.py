import random
import numpy as np
import pandas as pd
import yfinance as yf
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

timeframe = '1h'
ticker = 'GLD'
df = yf.download(tickers=ticker, period='1y', interval=timeframe)

def custom_mutate(individual, indpb=0.2):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] = random.randint(10, 30)
    return individual,

def hybrid_selection(population, size):
    intermediate_population = toolbox.selectTournament(population, 2 * size)
    final_population = toolbox.selectNSGA2(intermediate_population, size)
    return final_population

def moving_average(df, ma_period):
    df['MA'] = df['Adj Close'].rolling(window=ma_period).mean()
    return df

def strategy(df, ma_period):
    df['MA'] = df['Adj Close'].rolling(window=ma_period).mean()
    df['Signal'] = np.where(df['Adj Close'] > df['MA'], 1, -1)

    df['Prev Signal'] = df['Signal'].shift(1)
    df['Position'] = np.where(df['Signal'] != df['Prev Signal'], df['Signal'], np.nan)
    df['Position'] = df['Position'].ffill().fillna(0)

    df['Returns'] = df['Adj Close'].pct_change()
    df['Strategy Returns'] = df['Position'].shift(1) * df['Returns']

    df['Positive Returns'] = np.where(df['Strategy Returns'] > 0, df['Strategy Returns'], 0)
    df['Negative Returns'] = np.where(df['Strategy Returns'] < 0, df['Strategy Returns'], 0)

    total_positive_returns = df['Positive Returns'].sum()
    total_negative_returns = df['Negative Returns'].sum()

    return total_positive_returns, total_negative_returns

def evaluate(individual):
    ma_period = individual[0]
    df_with_ma = moving_average(df.copy(), ma_period)
    total_positive_returns, total_negative_returns = strategy(df_with_ma, ma_period)
    return total_positive_returns, total_negative_returns

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximize both positive and negative returns
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 30)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", custom_mutate)
toolbox.register("selectTournament", tools.selTournament, tournsize=3)
toolbox.register("selectNSGA2", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

NGEN = 50
CXPB = 0.9
MUTPB = 0.2
population = toolbox.population(n=500)

def evolve_population(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN):
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "std", "min", "max"

    for gen in range(1, ngen + 1):
        if keyboard.is_pressed('n'):
            print("Stopping the algorithm.")
            break
        
        offspring = hybrid_selection(population, len(population))

        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
        population = toolbox.selectNSGA2(population + offspring, len(population))
        
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
        print(logbook.stream)
        best_ind = tools.selBest(population, 1)[0]
        print(f"Gen {gen}: MA Period: {best_ind[0]}, Positive Returns: {best_ind.fitness.values[0]*100:.2f}%, Negative Returns: {best_ind.fitness.values[1]*100:.2f}%")

    return population, logbook

population, logbook = evolve_population(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN)

best_individual = tools.selBest(population, 1)[0]
ma_period = best_individual[0]
print(f"Best individual for {timeframe} timeframe:\n MA Period: {ma_period}\n Positive Returns: {best_individual.fitness.values[0]*100:.2f}%, Negative Returns: {best_individual.fitness.values[1]*100:.2f}")

# Plotting (optional)
# generations = range(1, len(logbook) + 1)
# avg_positive_returns = [log['avg'][0] for log in logbook]
# avg_negative_returns = [log['avg'][1] for log in logbook]
# max_positive_returns = [log['max'][0] for log in logbook]
# max_negative_returns = [log['max'][1] for log in logbook]
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# axs[0, 0].plot(generations, avg_positive_returns, label='Average Positive Returns', color='blue')
# axs[0, 0].set_title('Average Positive Returns Over Generations')
# axs[0, 0].set_xlabel('Generation')
# axs[0, 0].set_ylabel('Average Fitness Value')
# axs[0, 0].legend()
# axs[0, 0].grid(True)
# axs[0, 1].plot(generations, avg_negative_returns, label='Average Negative Returns', color='green')
# axs[0, 1].set_title('Average Negative Returns Over Generations')
# axs[0, 1].set_xlabel('Generation')
# axs[0, 1].set_ylabel('Average Fitness Value')
# axs[0, 1].legend()
# axs[0, 1].grid(True)
# axs[1, 0].plot(generations, max_positive_returns, label='Max Positive Returns', color='red')
# axs[1, 0].set_title('Max Positive Returns Over Generations')
# axs[1, 0].set_xlabel('Generation')
# axs[1, 0].set_ylabel('Maximum Fitness Value')
# axs[1, 0].legend()
# axs[1, 0].grid(True)
# axs[1, 1].plot(generations, max_negative_returns, label='Max Negative Returns', color='orange')
# axs[1, 1].set_title('Max Negative Returns Over Generations')
# axs[1, 1].set_xlabel('Generation')
# axs[1, 1].set_ylabel('Maximum Fitness Value')
# axs[1, 1].legend()
# axs[1, 1].grid(True)
# plt.tight_layout()
# plt.show()
