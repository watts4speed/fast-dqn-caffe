#!/usr/bin/env python
import matplotlib.pyplot as plt
import csv
import time
import os

def getColumn(filename, column):
    results = csv.reader(open(filename), delimiter=",")
    return [result[column] for result in results]

def plot_chart(chart_type, path_to_png, log_file):


    epoch = getColumn(log_file,0)
    score= getColumn(log_file,1)
    time=getColumn(log_file,2)
    head, rom = os.path.split(epoch[0]) #first row/column contains the rom file used
    steps_per_epoch = score[0]; # second column first row is the number of steps in a training epoch
    if len(time) <= 3:  # don't start plotting until there are 2 points to plot
        hours = "0"
    else:
        hours = str(time[-1]);

    title = "training " + rom + " for " + hours + " hours"
    


    plt.ion()
    plt.pause(0.01)
    plt.figure("Running Average Score")
    plt.xlabel("Epoch (" + steps_per_epoch + " training steps/epoch)")
    plt.ylabel("Score")
    plt.title(title)
        
    if len(epoch) > 3:
        plt.plot(epoch[2:], score[2:], linewidth = 2)
        plt.savefig(path_to_png)
        
    plt.show()
    plt.pause(15)



if __name__ == '__main__':
    while True:
        plot_chart(0, 'tran_progress.png', 'training_log.csv')
