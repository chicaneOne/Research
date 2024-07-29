import random
import numpy as np
import pandas as pd
import keyboard
import yfinance as yf
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

timeframe = '1h'
ticker = 'GLD'
df = yf.download(tickers=ticker, period='1y', interval=timeframe)

def generate_erc():
    return random.uniform(0,1)

def custom_mutate(individual, indpb=0.2):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            if i == 0:
                individual[i] = random.randint(10, 30)
            else:
                individual[i] = random.uniform(max(0, individual[i] - 0.1), min(1, individual[i] + 0.1))
    return individual,

def hybrid_selection(population, size):
    intermediate_population = toolbox.selectTournament(population, 2 * size)
    final_population = toolbox.selectNSGA2(intermediate_population, size)
    return final_population

def risk_adjusted_return(trades, risk_free_rate=0.02):
    if len(trades) == 0:
        return 0
    excess_returns = np.array(trades) - risk_free_rate / (252 * 6)
    downside_returns = excess_returns[excess_returns < 0]
    std_dev_downside = np.std(downside_returns)
    if std_dev_downside == 0:
        return 0
    return np.mean(excess_returns) / std_dev_downside


def rsi_calculation(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def trading_simulation(individual):
    rsi_period, erc_tp, erc_sl = individual
    local_df = df.copy()
    local_df['RSI'] = rsi_calculation(local_df['Adj Close'], rsi_period)
    
    trades = []
    position_open_price = []
    positionLimit = 3

    for i in range(2, len(local_df)):
        lastClosedRSI = local_df['RSI'].iloc[i-1]
        lastClosed2RSI = local_df['RSI'].iloc[i-2]
        current_price = local_df['Adj Close'].iloc[i]

        new_positions_open_price = []
        for price in position_open_price:
            trade_return = (current_price - price) / price
            if lastClosedRSI >= 70 or trade_return >= erc_tp / 100:
                trades.append(trade_return)
            elif trade_return <= -erc_sl / 100:
                trades.append(trade_return)
            else:
                new_positions_open_price.append(price)
        position_open_price = new_positions_open_price

        if lastClosed2RSI > 30 and lastClosedRSI <= 30 and len(position_open_price) < positionLimit:
            position_open_price.append(current_price)

    for price in position_open_price:
        trade_return = (current_price - price) / price
        trades.append(trade_return)

    equity_curve = np.cumsum(trades)
    return trades, equity_curve

def evaluate(individual):
    trades, equity_curve = trading_simulation(individual)
    cumulative_return = np.sum(trades)
    risk_adj_ret = risk_adjusted_return(trades)
    
    return cumulative_return, risk_adj_ret

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 30)
toolbox.register("attr_take_profit", generate_erc)
toolbox.register("attr_stop_loss", generate_erc)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_take_profit, toolbox.attr_stop_loss), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("selectTournament", tools.selTournament, tournsize=3)
toolbox.register("selectNSGA2", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

NGEN = 50
CXPB = 0.95
MUTPB = 0.5
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
        print(f"Gen {gen}: RSI Period: {best_ind[0]}, Take-Profit: {best_ind[1]:.2f}%, Stop-Loss: {best_ind[2]:.2f}%")

    return population, logbook

population, logbook = evolve_population(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN)

best_individual = tools.selBest(population, 1)[0]
rsi_period, erc_tp, erc_sl = best_individual
print(f"Best individual for {timeframe} timeframe:\n RSI Period: {rsi_period}\n Take-Profit Percentage: {erc_tp:.2f}%\n Stop-Loss Percentage: {erc_sl:.2f}%\n Fitness: {best_individual.fitness.values[0]*100:.2f}%, {best_individual.fitness.values[1]:.2f}")

generations = range(1, len(logbook) + 1)

avg_cumulative_returns = [log['avg'][0] for log in logbook]
avg_risk_adjusted_returns = [log['avg'][1] for log in logbook]
max_cumulative_returns = [log['max'][0] for log in logbook]
max_risk_adjusted_returns = [log['max'][1] for log in logbook]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

axs[0, 0].plot(generations, avg_cumulative_returns, label='Average Cumulative Return', color='blue')
axs[0, 0].set_title('Average Cumulative Return Over Generations')
axs[0, 0].set_xlabel('Generation')
axs[0, 0].set_ylabel('Average Fitness Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(generations, avg_risk_adjusted_returns, label='Average Risk-Adjusted Return', color='green')
axs[0, 1].set_title('Average Risk-Adjusted Return Over Generations')
axs[0, 1].set_xlabel('Generation')
axs[0, 1].set_ylabel('Average Fitness Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(generations, max_cumulative_returns, label='Max Cumulative Return', color='red')
axs[1, 0].set_title('Max Cumulative Return Over Generations')
axs[1, 0].set_xlabel('Generation')
axs[1, 0].set_ylabel('Maximum Fitness Value')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(generations, max_risk_adjusted_returns, label='Max Risk-Adjusted Return', color='orange')
axs[1, 1].set_title('Max Risk-Adjusted Return Over Generations')
axs[1, 1].set_xlabel('Generation')
axs[1, 1].set_ylabel('Maximum Fitness Value')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()