#created by Lai WEI, modified by
#plot the rtt statistical plots

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt

def compute_cdf(rtts,bins):
    count, bins_count = np.histogram(rtts, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf,bins_count[1:]

def draw_th_cdf(status_list,fig_path):
    asize = 36
    bsize = 36
    lwidth = 4
    # set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    # set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    # set up the tick size
    ax.tick_params(pad=18, labelsize=bsize - 2)

    cache_root = 'cache\\th'
    for status in status_list:
        cache_path = os.path.join(cache_root, status + '.csv')
        print(cache_path)
        trace = pd.read_csv(cache_path)
        trace = trace[:3000]
        ths = trace["Length/bytes"]/(trace["End Time/s"]-trace["On Time/s"])
        print(ths)
        ths /= 1024*1024
        #ax.plot(ths)
        cdf, bins = compute_cdf(ths, 100)
        ax.plot(cdf, label=status, linewidth=lwidth)

    ax.set_xlabel('Throughput(MB/s)', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    ax.set_xlim(0,20)
    #ax.set_ylim(0, 500)
    plt.legend(loc='lower right', fontsize=asize - 10)
    plt.savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    platform = "pc"
    status_list = ['FPS', 'Racing', 'RPG', 'Run']
    fig_root = "figs"
    fig_name = "th_cdf_" + platform
    fig_path = os.path.join(fig_root,fig_name+".pdf")
    draw_th_cdf(status_list,fig_path)
    # pp.test_violin_plot()
