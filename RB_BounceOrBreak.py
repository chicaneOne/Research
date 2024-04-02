import random
import numpy as np
import pandas as pd
import keyboard
from deap import base, creator, tools, algorithms
import yfinance as yf
import alpha_vantage as av

# Choose data type and ticker
data_type = "com"  # Choose "forex" or "stock"
forex_pair = "EURUSD=X"
stock = "AAPL"
crypto = "BTC-USD"
com = "GLD"

# Load data based on the choice
if data_type == "forex":
    ticker = forex_pair
elif data_type == "stock":
    ticker = stock
elif data_type == "crypto":
    ticker = crypto
elif data_type == "com":
    ticker = com

df = yf.download(tickers=ticker, period='58d', interval='15m')

def moving_average_fitness(individual):
    return calculate_percentage_profit(individual[0]),

def calculate_percentage_profit(ma_period):
    local_df = df.copy()
    local_df['MA'] = local_df['Adj Close'].rolling(window=ma_period).mean()

    cash = 100000
    position = 0  # Represents the quantity of the stock; positive for long, negative for short
    position_type = 'none'  # Can be 'long', 'short', or 'none'

    for i in range(2, len(local_df)):  # Start from the third row to ensure the second latest candle is always considered
        # Current and previous refer to the logic based on the second latest candle
        current_price = local_df['Adj Close'].iloc[i-1]  # This is the second latest candle's close
        previous_price = local_df['Adj Close'].iloc[i-2]  # This is the third latest candle's close
        current_ma = local_df['MA'].iloc[i-1]  # MA corresponding to the second latest candle
        previous_ma = local_df['MA'].iloc[i-2]  # MA corresponding to the third latest candle
        current_low = local_df['Low'].iloc[i-1]  # Low of the second latest candle
        current_high = local_df['High'].iloc[i-1]  # High of the second latest candle

        # Buy Signal - Only enter if no position or holding a short position
        if previous_price < previous_ma and current_price > current_ma:
            if position_type in ['none', 'short']:  # No position or in a short position
                cash += position * current_price  # Close short position if any
                position = cash / current_price  # Go long
                cash = 0
                position_type = 'long'

        # Buy on Bounce - Only enter if no position or holding a short position
        elif previous_price > previous_ma and current_price > current_ma and current_low <= current_ma:
            if position_type in ['none', 'short']:  # No position or in a short position
                cash += position * current_price  # Close short position if any
                position = cash / current_price  # Go long
                cash = 0
                position_type = 'long'

        # Sell Signal - Only enter if no position or holding a long position
        if previous_price > previous_ma and current_price < current_ma:
            if position_type in ['none', 'long']:  # No position or in a long position
                cash += position * current_price  # Close long position if any
                position = - (cash / current_price)  # Go short
                cash = 0
                position_type = 'short'

        # Sell on Bounce - Only enter if no position or holding a long position
        elif previous_price < previous_ma and current_price < current_ma and current_high >= current_ma:
            if position_type in ['none', 'long']:  # No position or in a long position
                cash += position * current_price  # Close long position if any
                position = - (cash / current_price)  # Go short
                cash = 0
                position_type = 'short'

    # Calculate final value (either close the position or just calculate the value)
    if position_type == 'long':
        final_value = cash + (position * local_df['Adj Close'].iloc[-1])
    elif position_type == 'short':
        final_value = cash + (position * local_df['Adj Close'].iloc[-1])  # This will be negative since position is negative for shorts
    else:
        final_value = cash  # No open position

    profit_percentage = ((final_value - 100000) / 100000) * 100
    return profit_percentage

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 5, 50)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", moving_average_fitness)
toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def evolve_population(population):
    # Preserve the best individuals
    top_individuals = tools.selBest(population, 1)  # Adjust the number as needed

    # Apply mutation and crossover
    offspring = algorithms.varAnd(population, toolbox, cxpb=0, mutpb=0.2)

    # Evaluate the new individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    population[:] = toolbox.select(offspring, k=len(population) - len(top_individuals)) + top_individuals

population = toolbox.population(n=200)
NGEN = 100

for gen in range(NGEN):
    if keyboard.is_pressed('n'):
        print("Stopping the algorithm.")
        break

    evolve_population(population)
    best_individual = tools.selBest(population, 1)[0]
    print(f"Generation {gen}: Best MA period {best_individual[0]}, Fitness {best_individual.fitness.values[0]}")

best_ma_setting = tools.selBest(population, 1)[0][0]
profit_percentage = calculate_percentage_profit(best_ma_setting)
print(f"Best overall MA period: {best_ma_setting} with a profit percentage of {profit_percentage}%")
