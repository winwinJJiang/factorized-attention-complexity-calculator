import numpy as np
import matplotlib.pyplot as plt
from c import *


SCALES = {'': 1e0, 'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12, 'P': 1e15}
UNITS = {'Memory': 'B', 'Computation': 'MACC'}
COLORS = {'Memory': 'SkyBlue', 'Computation': 'IndianRed'}


def draw_versus(mems, mem_scale, comps, comp_scale):
    mems = np.array(mems) / SCALES[mem_scale]
    comps = np.array(comps) / SCALES[comp_scale]
    ind = np.arange(len(mems)) / 2
    width = 0.2

    figure, mem_axis = plt.subplots(figsize=(4.5, 4.5))
    plt.yscale('symlog', subsy=[2, 5])
    rects1 = mem_axis.bar(
        ind - width / 2, mems, width, color='SkyBlue', label='Memory'
    )
    rects2 = mem_axis.bar(
        ind + width / 2, comps, width, color='IndianRed', label='Computation'
    )

    mem_axis.set_ylabel(
        f'Memory ({mem_scale}B) / Computation ({comp_scale}FLOPS)'
    )
    mem_axis.set_xticks(ind)
    mem_axis.set_xticklabels(['Factorized', 'Conventional'])
    mem_axis.legend()


def draw_complexity(data, scale, label):
    data = np.array(data) / SCALES[scale]
    ind = np.arange(len(data)) / 2
    width = 0.2

    figure, axis = plt.subplots(figsize=(5, 4.5))
    plt.yscale('log', subsy=[2, 5])
    rects1 = axis.bar(
        ind, data, width, color=COLORS[label], label=label
    )

    axis.set_ylabel(
        f'{label} ({scale}{UNITS[label]})'
    )
    axis.set_xticks(ind)
    axis.set_xticklabels([
        'F. 64x64', 'C. 64x64', 'F. 256x256', 'C. 256x256'
    ])
    axis.legend()


def draw_groups(facts, convs, scale, label):
    facts = np.array(facts) / SCALES[scale]
    convs = np.array(convs) / SCALES[scale]
    ind = np.arange(len(facts)) / 2
    width = 0.2

    figure, axis = plt.subplots(figsize=(5, 4.5))
    plt.yscale('log', subsy=[2, 5])
    axis.bar(
        ind - width / 2, facts, width, color='SkyBlue', label='Factorized'
    )
    axis.bar(
        ind + width / 2, convs, width, color='IndianRed', label='Conventional'
    )

    min_ = min(facts.min(), convs.min())
    max_ = max(facts.max(), convs.max())

    axis.set_ylim(min_ / 5, max_ * 5)
    axis.set_ylabel(
        f'{label} ({scale}{UNITS[label]})'
    )
    axis.set_xticks(ind)
    axis.set_xticklabels(['64x64', '256x256'])
    axis.legend()


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw_group(ss, facts, convs, scale, label, x_label, tick_frequency=1):
    facts = np.array(facts) / SCALES[scale]
    convs = np.array(convs) / SCALES[scale]
    indices = np.arange(len(ss))
    ss = np.array(ss)
    width = 0.2
    
    figure, axis = plt.subplots(figsize=(5, 4.5))

    plt.yscale('log')
    #plt.xscale('log', subsx=[])
    print(facts)
    axis.bar(indices - width / 2, facts, width, color='SkyBlue', label='Factorized')
    axis.bar(indices + width / 2, convs, width, color='IndianRed', label='Conventional')

    min_ = min(facts.min(), convs.min())
    max_ = max(facts.max(), convs.max())
    extension = 0.1
    axis.set_ylim(min_ * 0.1, max_ * (1 + extension))
    axis.set_ylabel(f'{label} ({scale}{UNITS[label]})')
    axis.set_xlabel(x_label)
    ticks = [tick for i, tick in enumerate(indices) if i % tick_frequency == 0]
    axis.set_xticks(indices)
    axis.set_xticklabels([str(ss[tick]) for tick in ticks])
    axis.legend()


def plot_group(ss, facts, convs, scale, label, x_label, tick_frequency=1):
    facts = np.array(facts) / SCALES[scale]
    convs = np.array(convs) / SCALES[scale]
    ss = np.array(ss)
    
    figure, axis = plt.subplots(figsize=(5, 4.5))

    #plt.yscale('log')
    plt.xscale('log', subsx=[])
    axis.plot(ss, facts, color='SkyBlue', label='Factorized')
    axis.plot(ss, convs, color='IndianRed', label='Conventional')

    min_ = min(facts.min(), convs.min())
    max_ = max(facts.max(), convs.max())

    extension = 0.1
    axis.set_ylim(min_ - (max_ * extension), max_ * (1 + extension))
    axis.set_ylabel(f'{label} ({scale}{UNITS[label]})')
    axis.set_xlabel(x_label)
    ticks = [s for i, s in enumerate(ss) if i % tick_frequency == 0]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(s) for s in ticks])
    axis.legend()


ss = [2 ** i for i in range(6, 9)]

fs = [fa_dot(s=s ** 2, c=64) for s in ss]
cs = [conv_dot(s=s ** 2, c=64) for s in ss]

fms = [complexity.memory for complexity in fs]
fcs = [complexity.compute for complexity in fs]
cms = [complexity.memory for complexity in cs]
ccs = [complexity.compute for complexity in cs]

draw_group(
    ss, fcs, ccs, '', 'Computation', 'Input side length', tick_frequency=1
)

plt.savefig("graph.png", dpi=400)
