# Step 1: Check that our project tools are available.
import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib

print("Python:", sys.version.split()[0])
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Setup successful! We are ready to explore the dataset.")
