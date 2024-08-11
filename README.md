# Multi-Objective Optimization and Genetic Programming of Technical Indicators for Trading

This research studies the integration of Multi-Objective Optimization (MOO) and Genetic Programming (GP) in trading strategies using Relative Strength Index (RSI) and Slow Stochastic (SS) indicators. The MOO agent in our method is the niched-Pareto Genetic Algorithm (NSGA-II). Our research attempts to minimize risk-adjusted returns and maximize cumulative returns by applying a multi-objective approach. Comparing the use of MOO and GP to conventional single-objective techniques, empirical data show a considerable improvement in strategy performance. Our results highlight how financial trading techniques can be improved by combining MOO and GP, possibly yielding larger returns at levels of controlled risk.

The file included in this research are: rsi.py and ss.py, while ma.py is a extended research from the initial research.

Results:
Analysing from the results, we can see that the indicators perform opposite results. RSI-based results a higher average cumulative return than SS-based, 20.8% to 14.56%. On the contrary, RSI-based average RAR(risk-adjusted return) result is much lower than SS-based, 1.07 to 10.22. The difference almost reached 1000%. Here, we find that SS has a greatly more downside return. We suspect this is caused by the negative result of risk-reward (tp = 0.58%, sl = 0.71%), producing more chance and space for big drawdown.
