from sort_trajectory_outcomes import *


trajectory_directory = '/Users/ec18006/OneDrive - University of Bristol/CHAMPS/Research_Topics/cyclopropylidene_bifurcation/including_N2/gas_phase_MD/progdyn_trajectories_right_leaning/archive'

index_list = ((0, 1), (2, 13), (5, 0, 1, 6))
stop_criteria_dict = {}

# Loop through all trajectories
trajectory_names, record = generate_data_table_all_trajs(trajectory_directory, index_list)

# Store information about product made and timing (trajectory name, forward product made, time made, backward product made, time made)
products_df = pd.DataFrame(record, columns=['forward', 'backward'])
results_df = pd.DataFrame(trajectory_names, columns=['name'])
results_df[['forward product', 'forward time']] = products_df['forward'].apply(pd.Series)
results_df[['backward product', 'backward time']] = products_df['backward'].apply(pd.Series)

# Make "label" column in dataframe to be able to sort by outcome
results_df['label'] = ""

for i in range(len(results_df)):
    if results_df['forward product'][i] == 'Diazocyclopropane' and results_df['backward product'][i] == 'R Allene':
        results_df.loc[i, 'label'] = 'R_to_diazo'

    elif results_df['backward product'][i] == 'Diazocyclopropane' and results_df['forward product'][i] == 'R Allene':
        results_df.loc[i, 'label'] = 'diazo_to_R'

    elif results_df['forward product'][i] == 'Diazocyclopropane' and results_df['backward product'][i] == 'S Allene':
        results_df.loc[i, 'label'] = 'S_to_diazo'

    elif results_df['backward product'][i] == 'Diazocyclopropane' and results_df['forward product'][i] == 'S Allene':
        results_df.loc[i, 'label'] = 'diazo_to_S'

    elif results_df['forward product'][i] == 'R Allene' and results_df['backward product'][i] == 'S Allene':
        results_df.loc[i, 'label'] = 'S_to_R'

    elif results_df['backward product'][i] == 'R Allene' and results_df['forward product'][i] == 'S Allene':
        results_df.loc[i, 'label'] = 'R_to_S'

    elif (results_df['forward product'][i] == 'Diazocyclopropane' and results_df['backward product'][i] == 'Diazocyclopropane'):
        results_df.loc[i, 'label'] = 'diazo_to_diazo'

    elif (results_df['forward product'][i] == 'S Allene' and results_df['backward product'][i] == 'S Allene'):
        results_df.loc[i, 'label'] = 'S_to_S'

    elif (results_df['forward product'][i] == 'R Allene' and results_df['backward product'][i] == 'R Allene'):
        results_df.loc[i, 'label'] = 'R_to_R'
    else:
        results_df.loc[i, 'label'] = 'incomplete'

print(results_df.sort_values(by=['label']).to_string())

sort_trajectories_into_folders(trajectory_directory, results_df)

# Get counts and average timing information
print(results_df.groupby('label').count()[['name']])
print(results_df.groupby('label').mean()[['forward time']])
print(results_df.groupby('label').mean()[['backward time']])
print(results_df.groupby('label')[['name']])

