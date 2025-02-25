"""
    Copyright (C) 2022 Marco Cominelli <marco.cominelli@unibs.it>
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
from optimization_utility import *
from os import listdir
import pickle
from plots_utility import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('all_dir', help='All the files in the directory, default no', type=int, default=0)
    parser.add_argument('name', help='Name of experiment file')
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('start_r', help='Start processing', type=int)
    parser.add_argument('end_r', help='End processing', type=int)
    parser.add_argument('bandwidth', help='bandwidth of the signal', type=int, choices={20, 40, 80, 160})
    parser.add_argument('wifi', help='wifi standard', choices={'ac', 'ax'})
    args = parser.parse_args()

    exp_save_dir = args.dir
    bandwidth = args.bandwidth
    wifi = args.wifi
    names = []

    if args.all_dir:
        all_files = listdir(exp_save_dir)
        mat_files = []
        for i in range(len(all_files)):
            if all_files[i].endswith('.mat'):
                names.append(all_files[i][:-4])
    else:
        names.append(args.name)

    for name in names:
        name_file = './phase_processing/signal_' + name + '.txt'
        with open(name_file, "rb") as fp:  # Pickling
            signal_complete = pickle.load(fp)

         # 802.11ac
        if wifi == 'ac':
            delta_f = 312.5E3
            if bandwidth == 20:
                delete_idxs = np.asarray([0, 1, 2, 3, 32, 61, 62, 63], dtype=int)
                pilot_subcarriers = [11, 25, 39, 53]
                F_frequency = 64
            elif bandwidth == 40:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 63, 64, 65, 123, 124, 125, 126, 127], dtype=int)
                pilot_subcarriers = [11, 39, 53, 75, 89, 117]
                F_frequency = 128
            elif bandwidth == 80:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)
                pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
                F_frequency = 256
            elif bandwidth == 160:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 383, 384, 385, 507, 508, 509, 510, 511], dtype=int)
                pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231, 261, 309, 345, 373, 395, 423, 459, 487]
                F_frequency = 512
        # 802.11ax
        elif wifi == 'ax':
            delta_f = 78.125E3
            if bandwidth == 20:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 125, 126, 127, 128, 129, 130, 131, 251, 252, 253, 254, 255], dtype=int)
                pilot_subcarriers = [12, 26, 38, 52, 66, 80, 92, 106, 118, 138, 150, 164, 176, 190, 204, 218, 230, 244]
                F_frequency = 256
            elif bandwidth == 40:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 254, 255, 256, 257, 258, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511], dtype=int)
                pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
                F_frequency = 512
            elif bandwidth == 80:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 509, 510, 511, 512, 513, 514, 515, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023], dtype=int)
                pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
                F_frequency = 1024
            elif bandwidth == 160:
                delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 509, 510, 511, 512, 513, 514, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1535, 1536, 1537, 1538, 1539, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047], dtype=int)
                pilot_subcarriers = [18, 32, 44, 58, 72, 86, 98, 112, 126, 140, 152, 166, 178, 192, 206, 220, 232, 246, 260, 274, 286, 300, 314, 328, 340, 354, 368, 382, 394, 408, 420, 434, 448, 462, 474, 488, 502, 522, 536, 550, 562, 576, 590, 604, 616, 630, 642, 656, 670, 684, 696, 710, 724, 738, 750, 764,  778, 792, 804, 818, 832, 846, 858, 872, 884, 898, 912,  26, 938, 952, 966, 980, 992, 1006, 1042, 1056, 1068, 1082, 1096, 1110, 1122, 1136, 1150, 1164, 1176, 1190, 1202, 1216, 1230, 1244, 1256, 1270, 1284, 1298, 1310, 1324, 1338, 1352, 1364, 1378, 1392, 1406, 1418, 1432, 1444, 1458, 1472, 1486, 1498, 1512, 1526, 1546, 1560, 1574, 1586, 1600, 1614, 1628, 1640, 1654, 1666, 1680, 1694, 1708, 1720, 1734, 1748, 1762, 1774, 1788, 1802, 1816, 1828, 1842, 1856, 1870, 1882, 1896, 1908, 1922, 1936, 1950, 1962, 1976, 1990, 2004, 2016, 2030]
                F_frequency = 2048

        subcarriers_space = 2
        delta_t = 1E-7
        delta_t_refined = 5E-9
        range_refined_up = 2.5E-7
        range_refined_down = 2E-7

        start_r = args.start_r
        if args.end_r != -1:
            end_r = args.end_r
        else:
            end_r = signal_complete.shape[1]

        frequency_vector_complete = np.zeros(F_frequency, )
        F_frequency_2 = F_frequency // 2
        for row in range(F_frequency_2):
            freq_n = delta_f * (row - F_frequency / 2)
            frequency_vector_complete[row] = freq_n
            freq_p = delta_f * row
            frequency_vector_complete[row + F_frequency_2] = freq_p
        frequency_vector = np.delete(frequency_vector_complete, delete_idxs)

        T = 1/delta_f
        t_min = -3E-7
        t_max = 5E-7

        T_matrix, time_matrix = build_T_matrix(frequency_vector, delta_t, t_min, t_max)
        r_length = int((t_max - t_min) / delta_t_refined)

        start_subcarrier = 0
        end_subcarrier = frequency_vector.shape[0]
        select_subcarriers = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)

        n_ss = args.nss
        n_core = args.ncore
        n_tot = n_ss * n_core

        # Auxiliary data for first step
        row_T = int(T_matrix.shape[0] / subcarriers_space)
        col_T = T_matrix.shape[1]
        m = 2 * row_T
        n = 2 * col_T
        In = scipy.sparse.eye(n)
        Im = scipy.sparse.eye(m)
        On = scipy.sparse.csc_matrix((n, n))
        Onm = scipy.sparse.csc_matrix((n, m))
        P = scipy.sparse.block_diag([On, Im, On], format='csc')
        q = np.zeros(2 * n + m)
        A2 = scipy.sparse.hstack([In, Onm, -In])
        A3 = scipy.sparse.hstack([In, Onm, In])
        ones_n_matr = np.ones(n)
        zeros_n_matr = np.zeros(n)
        zeros_nm_matr = np.zeros(n + m)

        for stream in range(0, 4):
            name_file = './phase_processing/r_vector' + name + '_stream_' + str(stream) + '.txt'
            signal_considered = signal_complete[:, start_r:end_r, stream]
            r_optim = np.zeros((r_length, end_r - start_r), dtype=complex)
            Tr_matrix = np.zeros((frequency_vector_complete.shape[0], end_r - start_r), dtype=complex)

            for time_step in range(end_r - start_r):
                signal_time = signal_considered[:, time_step]
                complex_opt_r = lasso_regression_osqp_fast(signal_time, T_matrix, select_subcarriers, row_T, col_T,
                                                           Im, Onm, P, q, A2, A3, ones_n_matr, zeros_n_matr,
                                                           zeros_nm_matr)

                position_max_r = np.argmax(abs(complex_opt_r))
                time_max_r = time_matrix[position_max_r]

                T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector, delta_t_refined,
                                                                       max(time_max_r - range_refined_down, t_min),
                                                                       min(time_max_r + range_refined_up, t_max))

                # Auxiliary data for second step
                col_T_refined = T_matrix_refined.shape[1]
                n_refined = 2 * col_T_refined
                In_refined = scipy.sparse.eye(n_refined)
                On_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
                Onm_refined = scipy.sparse.csc_matrix((n_refined, m))
                P_refined = scipy.sparse.block_diag([On_refined, Im, On_refined], format='csc')
                q_refined = np.zeros(2 * n_refined + m)
                A2_refined = scipy.sparse.hstack([In_refined, Onm_refined, -In_refined])
                A3_refined = scipy.sparse.hstack([In_refined, Onm_refined, In_refined])
                ones_n_matr_refined = np.ones(n_refined)
                zeros_n_matr_refined = np.zeros(n_refined)
                zeros_nm_matr_refined = np.zeros(n_refined + m)

                complex_opt_r_refined = lasso_regression_osqp_fast(signal_time, T_matrix_refined, select_subcarriers,
                                                                   row_T, col_T_refined, Im, Onm_refined, P_refined,
                                                                   q_refined, A2_refined, A3_refined,
                                                                   ones_n_matr_refined, zeros_n_matr_refined,
                                                                   zeros_nm_matr_refined)

                position_max_r_refined = np.argmax(abs(complex_opt_r_refined))

                T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector_complete, delta_t_refined,
                                                                       max(time_max_r - range_refined_down, t_min),
                                                                       min(time_max_r + range_refined_up, t_max))

                Tr = np.multiply(T_matrix_refined, complex_opt_r_refined)

                Tr_sum = np.sum(Tr, axis=1)

                Trr = np.multiply(Tr, np.conj(Tr[:, position_max_r_refined:position_max_r_refined + 1]))
                Trr_sum = np.sum(Trr, axis=1)

                Tr_matrix[:, time_step] = Trr_sum
                time_max_r = time_matrix_refined[position_max_r_refined]

                start_r_opt = int((time_matrix_refined[0] - t_min)/delta_t_refined)
                end_r_opt = start_r_opt + complex_opt_r_refined.shape[0]
                r_optim[start_r_opt:end_r_opt, time_step] = complex_opt_r_refined

            name_file = './phase_processing/r_vector_' + name + '_stream_' + str(stream) + '.txt'
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(r_optim, fp)

            name_file = './phase_processing/Tr_vector_' + name + '_stream_' + str(stream) + '.txt'
            with open(name_file, "wb") as fp:  # Pickling
                pickle.dump(Tr_matrix, fp)
