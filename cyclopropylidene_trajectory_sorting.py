from sort_trajectory_outcomes import *

trajectory_directory = '/Users/ec18006/OneDrive - University of Bristol/CHAMPS/Research_Topics/cyclopropylidene_bifurcation/including_N2/gas_phase_MD/progdyn_trajectories_left_leaning/archive'

index_list = ((0, 1), (2, 13), (5, 0, 1, 6))
geometric_parameters_list = ('C0C1[x]', 'C2N13[x]', 'F5C0C1F6[x]')
stop_criteria = {('C0C1[x] <= 1.55 and C2N13[x] <= 1.35'): 'Diazocyclopropane',
                 ('C0C1[x] >= 2.50 and F5C0C1F6[x] <= -90.0'): 'R Allene',
                 ('C0C1[x] >= 2.50 and F5C0C1F6[x] >= 90.0'): 'S Allene'}

# Loop through all trajectories
trajectory_names, record = generate_data_table_all_trajs(trajectory_directory, index_list, geometric_parameters_list, stop_criteria)
results_df = organize_data_table(trajectory_names, record, stop_criteria)
print(results_df)
print(results_df.sort_values(by=['label']).to_string())

sort_trajectories_into_folders(trajectory_directory, results_df)

# Get counts and average timing information
print(results_df.groupby('label').count()[['name']])
print(results_df.groupby('label').mean()[['forward time']])
print(results_df.groupby('label').mean()[['backward time']])
print(results_df.groupby('label')[['name']])

