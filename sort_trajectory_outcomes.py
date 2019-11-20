import numpy as np
import pandas as pd
import os
import glob
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def read_xyz_file(path):
    """ Reads in an xyz file from path as a DataFrame. This DataFrame is then turned into a 3D array such that the
    dimensions are (number of points) X (number of atoms) X 3 (Cartesian coordinates). The system name (based on the
    filename), list of atoms in the system, and Cartesian coordinates are output.
    :param path: path to xyz file to be read
    :return extensionless_system_name: str
            atom_list: numpy array
            cartesians: numpy array
    """
    system_name = path_leaf(path)
    # print("File being read is: %s" % system_name)

    extensionless_system_name = os.path.splitext(system_name)[0]

    data = pd.read_csv(path, header=None, delim_whitespace=True, names=['atom', 'X', 'Y', 'Z'])
    n_atoms = int(data.loc[0][0])
    n_lines_per_frame = int(n_atoms + 2)

    data_array = np.array(data)

    data_reshape = np.reshape(data_array, (int(data_array.shape[0] / n_lines_per_frame), n_lines_per_frame,
                                           data_array.shape[1]))
    cartesians = data_reshape[:, 2::, 1::].astype(np.float)
    atom_list = data_reshape[0, 2::, 0]

    return extensionless_system_name, atom_list, cartesians


def calculate_distance(indexes, cartesians):
    """
    Generates array of distances for each trajectory using the Cartesians of that trajectory.
    :param indexes: list of atom indexes between which to calculate distance. Two ints long.
    :param cartesians: n x N x 3 array of Cartesian coordinates along the course of a trajectory.
    :return: distance: n x 1 array of specified distance along course of trajectory
    """
    coordinates = np.array(cartesians)
    distance = np.sqrt(np.sum((coordinates[:, indexes[0]] - coordinates[:, indexes[1]]) ** 2, axis=1))

    return distance


def calculate_angle(indexes, cartesians):
    """
    Calculates the angle between three atom indexes
    :param indexes: list of atom indexes between which to calculate the angle. Three ints long.
    :param cartesians: n x N x 3 array of Cartesian coordinates along the course of a trajectory.
    :return: angle: n x 1 array of specified angle along course of trajectory
    """
    angle_radians = np.arccos((-calculate_distance((indexes[0], indexes[2]), cartesians) ** 2 +
                               calculate_distance((indexes[0], indexes[1]), cartesians) ** 2 +
                               calculate_distance((indexes[1], indexes[2]), cartesians) ** 2) /
                              (2 * calculate_distance((indexes[0], indexes[1]), cartesians) *
                               calculate_distance((indexes[1], indexes[2]), cartesians)))

    angle_degrees = 180 * angle_radians / np.pi

    return angle_degrees


def calculate_dihedral(indexes, cartesians):
    """
    Calculates the dihedral angle between four atom indexes
    :param indexes: list of atom indexes between which to calculate the angle. Four ints long.
    :param cartesians: n x N x 3 array of Cartesian coordinates along the course of a trajectory.
    :return: dihedral: n x 1 array of specified dihedral angle along course of trajectory
    """
    B1 = cartesians[:, indexes[1]] - cartesians[:, indexes[0]]
    B2 = cartesians[:, indexes[2]] - cartesians[:, indexes[1]]
    B3 = cartesians[:, indexes[3]] - cartesians[:, indexes[2]]

    modB2 = np.sqrt((B2[:, 0] ** 2) + (B2[:, 1] ** 2) + (B2[:, 2] ** 2))

    yAx = modB2 * B1[:, 0]
    yAy = modB2 * B1[:, 1]
    yAz = modB2 * B1[:, 2]

    # CP2 is the crossproduct of B2 and B3
    CP2 = np.cross(B2, B3)

    termY = (yAx * CP2[:, 0]) + (yAy * CP2[:, 1]) + (yAz * CP2[:, 2])

    # CP is the crossproduct of B1 and B2
    CP = np.cross(B1, B2)

    termX = (CP[:, 0] * CP2[:, 0]) + (CP[:, 1] * CP2[:, 1]) + (CP[:, 2] * CP2[:, 2])

    dihedral = (180 / np.pi) * np.arctan2(termY, termX)

    return dihedral


def split_trajectory(cartesians):
    # Find frame where first frame's Cartesians are repeated
    tss_frame = None
    for j in range(1, len(cartesians)):
        if np.array_equal(cartesians[j], cartesians[0]):
            tss_frame = j
            # print("Trajectory is %s frames long" % len(cartesians))
            # print("Trajectory halves split at frame %s" % tss_frame)

    if tss_frame == None:
        forward_half_of_traj_cartesians = cartesians
        backward_half_of_traj_cartesians = np.array([])
    else:
        forward_half_of_traj_cartesians = cartesians[0:tss_frame]
        backward_half_of_traj_cartesians = cartesians[tss_frame:]
        # print("Forward size: ", forward_half_of_traj_cartesians.shape)
        # print("Backward size: ", backward_half_of_traj_cartesians.shape)

    return forward_half_of_traj_cartesians, backward_half_of_traj_cartesians


def get_geometric_criteria(forward_half_of_traj_cartesians, backward_half_of_traj_cartesians,
                           geometric_criteria_indexes):
    """
    Calculates specified geometric criteria along trajectory. Returns list of distances/angles/dihedrals for both the
    forward half and backward half of the trajectory.
    :param forward_half_of_traj_cartesians: array of cartesian coordinates, forward half of trajectory
    :param backward_half_of_traj_cartesians: array of cartesian coordinates, backward half of trajectory
    :param geometric_criteria_indexes: list of list of ints
    :return: geometric_criteria_forward, geometric_criteria_backward
    """
    geometric_criteria_forward = []
    geometric_criteria_backward = []

    for i in range(len(geometric_criteria_indexes)):

        for traj_half, geom_half in zip((forward_half_of_traj_cartesians, backward_half_of_traj_cartesians),
                                        (geometric_criteria_forward, geometric_criteria_backward)):

            if not traj_half.any():
                break

            else:
                if len(geometric_criteria_indexes[i]) == 2:
                    distance = calculate_distance((geometric_criteria_indexes[i][0],
                                                   geometric_criteria_indexes[i][1]),
                                                  traj_half)
                    geom_half.append(distance)
                    # print(distance)

                if len(geometric_criteria_indexes[i]) == 3:
                    angle = calculate_angle((geometric_criteria_indexes[i][0],
                                             geometric_criteria_indexes[i][1],
                                             geometric_criteria_indexes[i][2]),
                                            traj_half)
                    # print(angle)
                    geom_half.append(angle)

                if len(geometric_criteria_indexes[i]) == 4:
                    dihedral = calculate_dihedral((geometric_criteria_indexes[i][0],
                                                   geometric_criteria_indexes[i][1],
                                                   geometric_criteria_indexes[i][2],
                                                   geometric_criteria_indexes[i][3]),
                                                  traj_half)
                    # print(dihedral)
                    geom_half.append(dihedral)

    return geometric_criteria_forward, geometric_criteria_backward


def impose_stop_criteria(geoms_forward, geoms_backward, geometric_parameters_list, stop_criteria):
    """
    Determines if and when particular products are formed along each half of a trajectory by imposing specific stop
    criteria, provided as a list.
    :param geoms_forward: geometric criteria along forward half of trajectory
    :param geoms_backward: geometric criteria along backward half of trajectory
    :param geometric_parameters_list: list of the geometric parameters that were measured in geoms_forward and
    geoms_backward, in the order that they appear in the list
    :param stop_criteria: dictionary of stop criteria where the keys are the expressions to be evaluated and the values
    are the structure(s) that are generated
    :return: record_halves: a record of what was formed first in each trajectory half and when
    """

    new_criteria = []
    for b in range(len(list(stop_criteria.keys()))):
        criterion = list(stop_criteria.keys())[b]

        for a in range(len(geometric_parameters_list)):
            criterion = criterion.replace(geometric_parameters_list[a], 'geom_half[%s][x]' % a)

        new_criteria.append(criterion)

    # Determine which product was made in each trajectory half
    record_halves = []
    for direction, geom_half in zip(('forward', 'backward'), (geoms_forward, geoms_backward)):

        structure_formed = []
        time = []

        if len(geom_half) >= 1:
            for y in range(len(new_criteria)):
                for x in range(len(geom_half[0])):
                    if eval(new_criteria[y]):
                        structure_formed.append(list(stop_criteria.values())[y])
                        # print('%s Formed at point %s' % (list(stop_criteria.values())[y], x))
                        time.append(x)
                        break

        if len(structure_formed) >= 1:
            first_formed = structure_formed[0]
            time_first = time[0]
            record_halves.append((first_formed, time_first))

    return record_halves


def impose_specific_stop_criteria(geoms_forward, geoms_backward):
    """
    Determines if and when particular products are formed along each half of a trajectory by imposing specific stop
    criteria, provided as a list.
    :param geoms_forward: geometric criteria along forward half of trajectory
    :param geoms_backward: geomtric criteria along backward half of trajectory
    :return: record_halves: a record of what was formed first in each trajectory half and when
    """
    # Determine which product was made in each trajectory half
    record_halves = []
    for direction, geom_half in zip(('forward', 'backward'), (geoms_forward, geoms_backward)):

        structure_formed = []
        time = []

        if len(geom_half)>= 1:
            C0C1 = geom_half[0]
            C2N13 = geom_half[1]
            F5C0C1F6 = geom_half[2]

            for x in range(len(geom_half[0])):
                if C0C1[x] <= 1.55 and C2N13[x] <= 1.35:
                    structure_formed.append('Diazocyclopropane')
                    # print('Diazocyclopropane Formed at point %s' % x)
                    time.append(x)
                    break

            for x in range(len(geom_half[0])):
                if C0C1[x] >= 2.50 and F5C0C1F6[x] <= -90.0:
                    structure_formed.append('R Allene')
                    # print('R Allene Formed at point %s' % x)
                    time.append(x)
                    break

            for x in range(len(geom_half[0])):
                if C0C1[x] >= 2.50 and F5C0C1F6[x] >= 90.0:
                    structure_formed.append('S Allene')
                    # print('S Allene Formed at point %s' % x)
                    time.append(x)
                    break

        if len(structure_formed) >= 1:
            first_formed = structure_formed[0]
            time_first = time[0]
            record_halves.append((first_formed, time_first))

    return record_halves


def generate_data_table_all_trajs(trajectory_directory, index_list, geometric_parameters_list, stop_criteria):

    xyz_files = sorted(glob.glob(os.path.join(trajectory_directory, '*.xyz')))
    record = []
    trajectory_names = []
    for xyz_file in xyz_files:
        trajectory_name, atom_list, cartesians = read_xyz_file(xyz_file)

        # Split trajectory at TSS (forward in time half and backward in time half)
        forward_cartesians, backward_cartesians = split_trajectory(cartesians)

        # Calculate geometric criteria along each trajectory half
        geoms_forward, geoms_backward = get_geometric_criteria(forward_cartesians, backward_cartesians, index_list)

        # Determine what structure was generated first in each half of a trajectory and when it was formed
        # record_halves = impose_specific_stop_criteria(geoms_forward, geoms_backward)
        record_halves = impose_stop_criteria(geoms_forward, geoms_backward, geometric_parameters_list, stop_criteria)

        trajectory_names.append(trajectory_name)
        record.append(record_halves)

    return trajectory_names, record


def organize_data_table(trajectory_names, record, stop_criteria):
    """
    Organizes the data table of trajectory results to include column names and a "label" column that can be used to
    count total number of particular trajectory outcomes.
    :param trajectory_names: list of trajectory names
    :param record: data table of trajectory results
    :param stop_criteria: dictionary of stop criteria used to specify products made
    :return: results_df: organized dataframe of results
    """
    # Store information about product made and timing (trajectory name, forward product made, time made, backward
    # product made, time made)
    products_df = pd.DataFrame(record, columns=['forward', 'backward'])
    results_df = pd.DataFrame(trajectory_names, columns=['name'])
    results_df[['forward product', 'forward time']] = products_df['forward'].apply(pd.Series)
    results_df[['backward product', 'backward time']] = products_df['backward'].apply(pd.Series)

    # Make "label" column in dataframe to be able to sort by outcome
    results_df['label'] = ""

    for i in range(len(results_df)):
        for a in range(len(stop_criteria.values())):
            for b in range(len(stop_criteria.values())):
                if str(results_df['forward product'][i]) == str(list(stop_criteria.values())[a]) and str(
                        results_df['backward product'][i]) == str(list(stop_criteria.values())[b]):
                    results_df.loc[i, 'label'] = '%s to %s' % (
                    list(stop_criteria.values())[b], list(stop_criteria.values())[a])
            if results_df.loc[i, 'label'] == "":
                results_df.loc[i, 'label'] = 'incomplete'

    return results_df


def sort_trajectories_into_folders(trajectory_directory, results_df):

    grouped_by_type = results_df.groupby('label')

    for key in list(grouped_by_type.groups.keys()):
        if not os.path.exists(os.path.join(trajectory_directory, key.replace(" ", "_"))):
            os.makedirs(os.path.join(trajectory_directory, key.replace(" ", "_")))
        for traj_name in grouped_by_type.get_group(key)['name']:
            os.rename(os.path.join(trajectory_directory, traj_name + ".xyz"),
                      os.path.join(trajectory_directory, key.replace(" ", "_"), traj_name + ".xyz"))


