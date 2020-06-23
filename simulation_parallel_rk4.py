import numpy as np
import math
import time
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt

#做一个drive的oscillation
def Vel_Verlet_simu_decoupled(state, drive_freq, if_plot='True'):#state - dictionary, ....at one drive_freq
    mu = state['mu']/10# drag
    F = state['F']
    wsquare = state['wsquare']*10
    a = state['a']
    b = state['b']/1000


    def acceleration(x1, v1, count):#1-sloshing, 2-breathing
        acc_1 = F * math.sin(drive_freq * 2 * math.pi * count) - mu * v1 - wsquare * x1 - a * x1 * x1 - b * x1 * x1 * x1

        diverge = False
        if math.isnan(acc_1):
            diverge = True
        return acc_1, diverge

    #Velocity_Verlet_sim
    total_step = 100000
    dt = 0.001
    fps = 1 / dt
    count = 0
    # initialization
    position1 = np.zeros(total_step)
    vel1 = np.zeros(total_step)
    acc1 = np.zeros(total_step)

    #initial
    vel1[0]=1
    #RK4 simulation
    for i in range(total_step-1):
        diverge_total = False
        k11 = dt * vel1[i]
        acc1_temp1, diverge1 = acceleration(position1[i], vel1[i], count)
        if diverge1:
            diverge_total = True
            break
        k21 = dt * acc1_temp1
        k12 = dt * (vel1[i]+0.5*k21)
        acc1_temp2, diverge2 = acceleration(position1[i]+0.5*k11, vel1[i]+0.5*k21, count+0.5*dt)
        if diverge2:
            diverge_total = True
            break
        k22 = dt * acc1_temp2
        k13 = dt * (vel1[i] + 0.5*k22)
        acc1_temp3, diverge3 = acceleration(position1[i] + 0.5 * k12, vel1[i] + 0.5 * k22, count + 0.5 * dt)
        if diverge3:
            diverge_total = True
            break
        k23 = dt * acc1_temp3
        k14 = dt * (vel1[i] + k23)
        acc1_temp4, diverge4 = acceleration(position1[i] + k13, vel1[i] + k23, count + dt)
        if diverge4:
            diverge_total = True
            break
        k24 = dt * acc1_temp4
        position1[i+1] = position1[i] + (k11 + 2 * k12 + 2 * k13 + k14) / 6
        vel1[i+1] = vel1[i] + (k21 + 2 * k22 + 2 * k23 + k24) / 6
        count = count + dt
    if diverge_total:
        print('diverge')
        return 0, fps, diverge_total
    else:
        position1_stable = position1[math.floor(0.5 * total_step):]

        plt.figure()
        plt.plot(position1[:])

        return position1_stable, fps, diverge_total

def FFT_peak_search(p1, drive_freq, fps, if_plot='True'):#half the driveing freq, twice the driving freq
    search_width = 10
    frame = len(p1)

    x_axis_temp = [i for i in range(frame)]
    x_axis = np.array(x_axis_temp) * fps / frame
    I = np.argmin((x_axis - drive_freq) * (x_axis - drive_freq))
    half_I = np.argmin((x_axis - 0.5 * drive_freq) * (x_axis - 0.5 * drive_freq))
    twice_I = np.argmin((x_axis - 2 * drive_freq) * (x_axis - 2 * drive_freq))

    p1_fft_temp = np.fft.fft(p1)
    p1_fft = np.sqrt(p1_fft_temp * np.conj(p1_fft_temp)) / (0.5 * frame)
    primary_peak1 = np.max(p1_fft[I - search_width: I + search_width])
    half_peak1 = np.max(p1_fft[half_I - search_width: half_I + search_width])
    twice_peak1 = np.max(p1_fft[twice_I - search_width: twice_I + search_width])

    

    return [abs(primary_peak1), abs(half_peak1), abs(twice_peak1)]



def parallel_pack(state, drive_freq):#并行运算的pack函数
    p1, fps, diverge = Vel_Verlet_simu_decoupled(state, drive_freq, if_plot='False')
    if diverge:
        return (-1, [0, 0, 0])
    else:
        p1_peaks = FFT_peak_search(p1, drive_freq, fps, if_plot='False')
        return (drive_freq, p1_peaks)

def response_curve_multiprocess(state, drive_scan):
    drives = drive_scan

    result = []
    with concurrent.futures.ProcessPoolExecutor() as excutor:
        futures = [excutor.submit(parallel_pack, state, item) for item in drives]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result(35))
    result.sort()
    if result[0][0] == -1:
        #print('diverge')
        diverge = True
        return 0, 0, 0, diverge
    else:
        diverge = False
        drive, p1_resp = zip(*result)
        drive = np.array(drive)
        p1_resp = np.array(p1_resp)
        row = len(drive)
        primay_p1 = np.zeros((row, 2))
        primay_p1[:, 0] = drive
        primay_p1[:, 1] = p1_resp[:, 0]

        twice_p1 = np.zeros((row, 2))
        twice_p1[:, 0] = drive
        twice_p1[:, 1] = p1_resp[:, 2]

        half_p1 = np.zeros((row, 2))
        half_p1[:, 0] = drive
        half_p1[:, 1] = p1_resp[:, 1]


        return primay_p1, twice_p1, half_p1, diverge


#算给定state的所有MSE
def MSEs(state, drive_scan, exp1_p, exp1_s):#做所有的MSE..exp1_p=ndarray(1列freq2列response )
    primay_p1, twice_p1, half_p1, diverge = response_curve_multiprocess(state, drive_scan)
    
    if diverge:
        MSE1_p = 10000
        MSE1_s = 10000
    else:
        assert ((primay_p1[:, 0] == exp1_p[:, 0]).all())
        twice_p1_window = twice_p1[0: exp1_s.shape[0], :]
        
        #MSE-percentage diff squared
        MSE1_p = np.sum(((primay_p1[:, 1] - exp1_p[:, 1]) / exp1_p[:, 1]) * ((primay_p1[:, 1] - exp1_p[:, 1]) / exp1_p[:,1])) / primay_p1.shape[0]
        MSE1_s = np.sum(((twice_p1_window[:, 1] - exp1_s[:, 1]) / exp1_s[:, 1]) * ((twice_p1_window[:, 1] - exp1_s[:, 1]) / exp1_s[:, 1])) / twice_p1_window.shape[0]
    return MSE1_p, MSE1_s




#读入experiment_response
def read_experiment_responses():
    exp1_p = pd.read_csv('p1V.csv')
    exp1_s = pd.read_csv('s1V.csv')
    return exp1_p.values, exp1_s.values


def packed_fun(state, exp1_p, exp1_s):
    drive_scan = exp1_p[:, 0]
    mse1_p, mse1_s = MSEs(state, drive_scan, exp1_p, exp1_s)
    return mse1_p, mse1_s



if __name__ == '__main__':

    def inital():
        state = {'F': 62912.46757136506, 'a': -20.90626114515319, 'b': 137.69784620223381, 'mu': 101.09590040894685,
                 'wsquare': 559.6906658471319}
        position1_stable, fps, diverge_total = Vel_Verlet_simu_decoupled(state, 15, if_plot='True')
        return position1_stable, fps, diverge_total
    inital()







