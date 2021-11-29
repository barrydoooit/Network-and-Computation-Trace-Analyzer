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

def draw_rtt_cdf(status_list,fig_path):
    asize = 36
    bsize = 36
    lwidth = 5
    # set up the figure canvas
    figure = plt.figure(figsize=(16, 9), dpi=80)
    # set up the margin
    ax = figure.add_axes([0.115, 0.15, 0.8, 0.8])
    # set up the tick size
    ax.tick_params(pad=18, labelsize=bsize - 2)

    cache_root = 'minecraft_data/processed_data/rtt'
    pcapng_root = 'caps'
    for status in status_list:
        cache_path = os.path.join(cache_root, status + '_rtt.csv')
        print(cache_path)
        pcapng_path = os.path.join(pcapng_root, status + '.pcapng')
        print(pcapng_path)
        rtts = pd.read_csv(cache_path)
        cdf, bins = compute_cdf(rtts["sample_rtt"], 100)
        ax.plot(bins, cdf, label=status, linewidth=lwidth)

    ax.set_xlabel('RTT(ms)', fontsize=asize)
    ax.set_ylabel('CDF', fontsize=asize)
    #ax.set_xlim(4, 11)
    ax.set_ylim(0, 1)
    plt.legend(loc='lower right', fontsize=asize - 10)
    plt.savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    platform = "pc"
    status_list = ['connect','fast_chunk_load','fast_chunk_reload',
              'short_range_move','stan_still_with_creatures',
              'stand_still']
    fig_root = "figs"
    fig_name = "rtt_cdf_" + platform
    fig_path = os.path.join(fig_root,fig_name+".pdf")
    draw_rtt_cdf(status_list,fig_path)
    # pp.test_violin_plot()
