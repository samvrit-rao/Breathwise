import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Assuming patient_df is already loaded and structured as described
# Transforming patient_df to a suitable format for Kaplan-Meier analysis
patient_df = pd.read_csv('/Users/raosamvr/Downloads/tenyeardata.csv')

# Plotting the Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))
plt.ylim(0, 1)

for group in patient_df['group'].unique():
    group_data = patient_df[patient_df['exacerbation'] == group]
    kmf.fit(durations=group_data["year"], event_observed=group_data["death"], label=group)
    kmf.plot_survival_function(ci_show=False)

plt.rcParams['font.family'] = 'Arial'
plt.title('Kaplan-Meier Survival Curves', fontweight='bold')
plt.xlabel('Time (Years)', fontweight='bold')
plt.ylabel('Survival Rate', fontweight='bold')
plt.legend()
plt.show()
