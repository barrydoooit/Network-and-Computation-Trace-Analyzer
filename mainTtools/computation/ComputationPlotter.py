import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""import altair as alt
import altair_viewer"""
import matplotlib
matplotlib.style.use('ggplot')
DATA_ROOT = '../../data/computation/'


def plot_cpu_usage_distribution(data_dir='20211209_First/', games=('vrchat', 'minecraft',),
                                stages=('login', 'action', 'stay',)):
    stages_dfall = []
    data_path_base = DATA_ROOT + data_dir
    percentiles = (50, 75, 90, 95, 100)
    for s in stages:
        stages_dfall.append(pd.DataFrame(
            columns=["0-%d%%" % percentiles[0]] + ["%d%%-%d%%" % (percentiles[i], percentiles[i + 1]) for i in
                                                   range(len(percentiles) - 1)], index=games))
        for g in games:
            df = pd.read_csv('%s%s_data/%s_%s.csv' % (data_path_base, g, g, s))
            df.drop(columns=['Mem/MB'], inplace=True)
            percentile_diff = np.insert(np.diff([np.percentile(df['CPU/Percentage'], p) for p in percentiles]), 0, np.percentile(df['CPU/Percentage'], percentiles[0]))
            stages_dfall[-1].loc[g] = percentile_diff

    fig, axes = plt.subplots(nrows=1, ncols=len(stages), figsize=(8, 6), dpi=160)
    ax_position = 0
    for stage, stage_name in zip(stages_dfall, stages):
        ax = stage.plot(kind="bar", stacked=True, colormap="Blues_r",
                         ax=axes[ax_position])
        ax.set_title("Stage \"" + stage_name + "\"", alpha=0.8, fontsize=10)
        ax.set_ylabel("CPU Usage (%)"),
        # ax.set_yticks(range(0, , 50))
        #ax.set_yticklabels(labels=range(0, 9000, 1000), rotation=0, minor=False, fontsize=28)
        ax.set_xticklabels(labels=games, rotation=0, minor=False)
        handles, labels = ax.get_legend_handles_labels()
        ax_position += 1

    # look "three subplots"
    # plt.tight_layout(pad=0.0, w_pad=-8.0, h_pad=0.0)

    # look "one plot"
    for a in axes[:-1]:
        a.legend().set_visible(False)
    for a in axes[1:]:
        a.set_ylabel("")
        a.set_yticklabels("")
    axes[-1].legend(["0-%d%%" % percentiles[0]] + ["%d%%-%d%%" % (percentiles[i], percentiles[i + 1]) for i in
                                                   range(len(percentiles) - 1)], loc='upper right', fontsize=10)
    plt.xlabel('Games')
    plt.show()
    fig.savefig("../../figs/"+"-".join(stages)+"-".join(games)+"cpu_usage_percentile_distribution.pdf")
    """def prep_df(df, name):
        df = df.stack().reset_index()
        df.columns = ['c1', 'c2', 'values']
        df['DF'] = name
        return df
    names = ['DF1','DF2']
    df = pd.concat([prep_df(d, g) for d, g in zip(game_dfall, names)])
    alt.renderers.enable('html')
    alt.Chart(df).mark_bar().encode(
        # tell Altair which field to group columns on
        x=alt.X('c2:N', title=None),
        # tell Altair which field to use as Y values and how to calculate
        y=alt.Y('sum(values):Q',
                axis=alt.Axis(
                    grid=False,
                    title=None)),
        # tell Altair which field to use to use as the set of columns to be  represented in each group
        column=alt.Column('c1:N', title=None),
        # tell Altair which field to use for color segmentation
        color=alt.Color('DF:N',
                        scale=alt.Scale(
                            # make it look pretty with an enjoyable color pallet
                            range=['#96ceb4', '#ffcc5c', '#ff6f69'],
                        ),
                        )) \
        .configure_view(
        strokeOpacity=0
    ).show()"""

if __name__ == '__main__':
    plot_cpu_usage_distribution()
