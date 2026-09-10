#!/usr/bin/env python3
"""Render the article from aggregate snapshots; does not rerun experiments."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    snapshot = json.loads((ROOT / 'results/confusion.json').read_text())
    plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9})
    fig, axes = plt.subplots(1, len(snapshot['models']), figsize=(6.2, 3.0))
    for ax, model in zip(np.atleast_1d(axes), snapshot['models']):
        cm = np.array(model['matrix'])
        if cm.shape != (2, 2) or cm.sum() != snapshot['n_test']:
            raise ValueError('Invalid confusion snapshot')
        shade = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(shade, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}\n({shade[i,j]:.1%})', ha='center', va='center',
                        color='white' if shade[i,j] > .6 else '#202020')
        ax.set_xticks([0, 1], snapshot['labels'])
        ax.set_yticks([0, 1], snapshot['labels'])
        ax.set(xlabel='predicted', ylabel='actual', title=model['name'])
    fig.tight_layout()
    (ROOT / 'paper/figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'paper/figures/fig_confusion.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    with (ROOT / 'results/pilot.csv').open() as f:
        rows = list(csv.DictReader(f))
    tex = [r'\begin{table*}[t]',
           r'\caption{Historical pilot on NVIDIA Jetson Orin Nano. Board telemetry was sampled at 500 ms. These aggregate rows are not remeasured by the asset renderer.}',
           r'\label{tab:pilot}', r'\centering\small',
           r'\begin{tabular}{@{}lllrrr@{}}', r'\toprule',
           r'Model & Runtime & Precision & Median (ms) & Throughput (inf./s) & Marginal energy (mJ) \\',
           r'\midrule']
    for row in rows:
        cells = [row['model'], row['runtime'], row['precision'],
                 f"{float(row['median_ms']):.3f}", f"{float(row['throughput_inf_s']):,.0f}",
                 f"{float(row['marginal_energy_mj']):.2f}"]
        tex.append(' & '.join(cells) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    (ROOT / 'paper/sections/06_pilot_table.tex').write_text('\n'.join(tex))
    print('Rendered aggregate confusion figure and pilot table.')


if __name__ == '__main__':
    main()
