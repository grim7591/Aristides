# %% [markdown]
# # ACPA Property Data Model in Process documentation
# 
# **Initial steps**
# 
# - Figure out how to make a satisfying markdown document
# to be able to keep track of progress (Done)
# - Get data from PACs for model training
#   - Options:
#       - Use current year sales to train
#           - Probably not the move but might not be a bad place to start.
#       - Use time adjusted sales to train
#           - Time adjust using historical data, might not be too hard.
#           - One year of sales probably isn't enough because it biases the model towards 2023-like market conditions
#       - Use non time adjusted sales and year becomes a factor?
# - Identify less than 20 features to test for
#   - Examples from Lee:
#       - base area, age, 
#       - improvement code 
#       - quality, 
#       - land square ft
# - Do everything else
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 