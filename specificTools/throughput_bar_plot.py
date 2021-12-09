#created by Lai WEI, modified by
#plot the rtt statistical plots

import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from configs import BarChartConfigs

def th_stats(trace):
    ths = np.array(trace["Throughput(KBytes/s)"])
    avg = ths.mean()
    std = ths.var()
    return avg,std

def packet_split(trace):
    basic = trace[trace["Length/bytes"]>1000]
    messaging = trace[trace["Length/bytes"]<=500]
    addition = trace[trace["Length/bytes"]>=500][trace["Length/bytes"]<1000]
    return basic,addition,messaging

def compute_cdf(rtts,bins):
    count, bins_count = np.histogram(rtts, bins=bins)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf,bins_count[1:]

def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = 0
        if rect.get_y() < 0:
            height = rect.get_y()
        else:
            height = rect.get_height()

        print(rect.get_height())
        print( str(rect.get_y()) )

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)
        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width() / 2., label_position,
                '%.2f' % height,
                fontsize=24,
                ha='center', va='bottom')

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

    cache_root = 'cache/th'
    group = 0
    basics = []
    basics_var = []
    additions = []
    additions_var = []
    messagings = []
    messagings_var = []
    for status in status_list:
        cache_path = os.path.join(cache_root, status + '.csv')
        print(cache_path)
        trace = pd.read_csv(cache_path)
        trace = trace[:3000]
        (basic_trace, addition_trace, messaging_trace) = packet_split(trace)
        basic_stat = th_stats(basic_trace)
        addition_stat = th_stats(addition_trace)
        messaging_stat = th_stats(messaging_trace)
        print(basic_stat[0])
        basics.append(basic_stat[0])
        additions.append(addition_stat[0])
        messagings.append(messaging_stat[0])
        basics_var.append(basic_stat[1]/100)
        additions_var.append(addition_stat[1]/100)
        messagings_var.append(messaging_stat[1]/100)

    offset = BarChartConfigs.bar_width
    basic_pos = np.arange(len(basics)) * BarChartConfigs.bar_interval
    rects_basic = ax.bar(basic_pos, basics, label='Basic Streaming', color='white',
                          yerr=basics_var,
                         hatch="//",linewidth=lwidth,
                         edgecolor='red',ecolor='black', capsize=10,
                         align='center', alpha=BarChartConfigs.opaque,
                         width=BarChartConfigs.bar_width)
    rectsx_basic = ax.bar(basic_pos, basics,
                         yerr=basics_var,
                         color='none', edgecolor='black', linewidth=lwidth,
                         align='center', width=BarChartConfigs.bar_width)
    basic_ticks = BarChartConfigs.abalation_baloss_ticks

    addition_pos = np.arange(len(additions)) * BarChartConfigs.bar_interval + offset * 1
    rects_addition = ax.bar(addition_pos, additions, label='Addition Streaming', color='white',
                         yerr=additions_var,
                         hatch="x",
                         edgecolor="blue",ecolor='black', capsize=10,
                         align='center', alpha=BarChartConfigs.opaque,
                         width=BarChartConfigs.bar_width)
    rectsx_addition = ax.bar(addition_pos, additions,
                          color='none', edgecolor='black', linewidth=lwidth,
                          align='center', width=BarChartConfigs.bar_width)

    messaging_pos = np.arange(len(messagings)) * BarChartConfigs.bar_interval + offset * 2
    rects_messaging = ax.bar(messaging_pos, messagings, label='Messaging', color='white',
                            yerr=messagings_var,
                            hatch="+",
                            edgecolor="green",ecolor='black', capsize=10,
                            align='center', alpha=BarChartConfigs.opaque,
                            width=BarChartConfigs.bar_width)
    rectsx_messaging = ax.bar(messaging_pos, messagings,
                             color='none', edgecolor='black', linewidth=lwidth,
                             align='center', width=BarChartConfigs.bar_width)
    plt.xticks(basic_pos+offset, basic_ticks)
    autolabel(rects_basic, ax)
    autolabel(rects_addition, ax)
    autolabel(rects_messaging, ax)
    ax.set_ylabel('Throughput(MB/s)', fontsize=asize)
    plt.legend(loc='upper right', fontsize=asize - 10)
    plt.savefig(fig_path)
    plt.show()

if __name__ == "__main__":
    platform = "pc"
    status_list = ['FPS', 'Racing', 'RPG', 'Run']
    fig_root = "figs"
    fig_name = "th_bar_" + platform
    fig_path = os.path.join(fig_root,fig_name+".pdf")
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 3
    plt.rcParams["legend.handlelength"] = 1.0
    draw_th_cdf(status_list,fig_path)
    # pp.test_violin_plot()
