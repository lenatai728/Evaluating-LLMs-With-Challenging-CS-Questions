#!/usr/bin/env python3
"""
6_visualize.py — Generate publication-quality graphs from model evaluation CSVs.

Usage:
    python3 6_visualize.py --input_files file1.csv file2.csv ...
    python3 6_visualize.py --input_files data/4_final_results/*_result_by_model.csv

Outputs 14 PNG graphs to data/5_graphs/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join('data', '5_graphs')

# Model parameter sizes (billions) for scaling-law chart
PARAM_SIZES = {
    'gemma-3-1b-it':  1,
    'gemma-3-4b-it':  4,
    'gemma-3-12b-it': 12,
    'gemma-3-27b-it': 27,
    'deepseek-chat':  671,
}

# Per-model colour palette (consistent across all charts)
MODEL_COLORS = {
    'gemma-3-1b-it':  '#6BAED6',
    'gemma-3-4b-it':  '#3182BD',
    'gemma-3-12b-it': '#08519C',
    'gemma-3-27b-it': '#082F6C',
    'deepseek-chat':  '#E6550D',
}
DEFAULT_COLOR = '#999999'

DOMAIN_LABELS = ['DSA', 'CSA', 'DIS', 'SEAI', 'Cyber']
DOMAIN_FULL   = {
    'DSA':   'Data Structures\n& Algorithms',
    'CSA':   'Computer Systems\n& Architecture',
    'DIS':   'Databases &\nInfo Systems',
    'SEAI':  'Software Eng\n& AI Fundamentals',
    'Cyber': 'Cybersecurity',
}
QTYPE_LABELS = ['MC', 'MS', 'TF', 'FB', 'OE']
QTYPE_FULL = {
    'MC': 'Multiple\nChoice',
    'MS': 'Multiple\nSelect',
    'TF': 'True/False',
    'FB': 'Fill-in-\nBlank',
    'OE': 'Open-\nEnded',
}


# ─── STYLE SETUP ─────────────────────────────────────────────────────────────

def setup_style():
    """White professional theme"""
    plt.rcParams.update({
        'figure.facecolor':   'white',
        'axes.facecolor':     'white',
        'savefig.facecolor':  'white',
        'axes.edgecolor':     '#333333',
        'axes.labelcolor':    '#222222',
        'text.color':         '#222222',
        'xtick.color':        '#333333',
        'ytick.color':        '#333333',
        'axes.grid':          False,
        'grid.color':         '#E0E0E0',
        'grid.linewidth':     0.5,
        'font.size':          11,
        'axes.titlesize':     14,
        'axes.labelsize':     12,
        'legend.fontsize':    10,
        'figure.dpi':         300,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'font.family':        'sans-serif',
    })


def get_color(model_name):
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {filename}")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def sort_by_score(df):
    """Return df sorted by Final Model Score descending."""
    return df.sort_values('Final Model Score', ascending=False).reset_index(drop=True)


def short_name(model):
    """Shorten model name for chart labels."""
    return model
    # return model.replace('gemma-3-', 'Gemma3-').replace('deepseek-chat', 'DeepSeek-V3')


# ─── SECTION A: DATASET CONSTRUCTION ────────────────────────────────────────

def plot_01_question_types_pie(df):
    """Donut chart: Question types."""
    row = df.iloc[0]
    labels = ['MC', 'MS', 'TF', 'FB', 'OE']
    sizes = [row[f'{t} Count'] for t in labels]
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78, textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color('white')
        t.set_fontweight('bold')

    # Center text
    ax.text(0, 0, f'{int(total)}', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#333333')
    ax.text(0, -0.08, 'Questions', ha='center', va='top',
            fontsize=12, color='#666666')

    ax.set_title('Distribution of Question Types', fontsize=16, fontweight='bold', pad=20)

    legend_labels = [f'{l}: {int(s)} questions' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10, frameon=False)
    save(fig, '01_question_types_pie.png')


def plot_02_domains_pie(df):
    """Donut chart: Domains."""
    row = df.iloc[0]
    sizes = [row[f'{d} Count'] for d in DOMAIN_LABELS]
    labels_display = ['DSA', 'CSA', 'DIS', 'SEAI', 'Cyber']
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_display, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78, textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_color('white')
        t.set_fontweight('bold')

    # Center text
    ax.text(0, 0, f'{int(total)}', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#333333')
    ax.text(0, -0.08, 'Questions', ha='center', va='top',
            fontsize=12, color='#666666')

    ax.set_title('Distribution of Question Domains', fontsize=16, fontweight='bold', pad=20)

    legend_labels = [f'{l}: {int(s)} questions' for l, s in zip(labels_display, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10, frameon=False)
    save(fig, '02_domains_pie.png')


def plot_03_calc_pie(df):
    """Donut chart: Calculation vs Non-Calculation."""
    row = df.iloc[0]
    sizes = [row['Calc Count'], row['Non-Calc Count']]
    labels = ['Calculation', 'Non-Calculation']
    colors = ['#4E79A7', '#F28E2B']
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78, textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_color('white')
        t.set_fontweight('bold')

    # Center text
    ax.text(0, 0, f'{int(total)}', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#333333')
    ax.text(0, -0.08, 'Questions', ha='center', va='top',
            fontsize=12, color='#666666')

    ax.set_title('Calculation vs Non-Calculation Questions', fontsize=16, fontweight='bold', pad=20)

    legend_labels = [f'{l}: {int(s)} questions' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=11, frameon=False)
    save(fig, '03_calc_vs_noncalc_pie.png')

# ─── SECTION B: CORE PERFORMANCE ────────────────────────────────────────────

def plot_04_model_score_bar(df):
    """Bar chart: Final Model Score by model."""

    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    scores = df_s['Final Model Score'].values
    colors = [get_color(m) for m in df_s['Model Name']]

    # make models on y-axis and score on x-axis
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, scores, color=colors, height=0.6, edgecolor='white', linewidth=0.5)

    # put percentage labels outside the bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{score:.1f}%', ha='left', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Final Model Score (%)')
    ax.set_title('Q→AR: Final Model Score by Model', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '04_model_score_bar.png')

    # fig, ax = plt.subplots(figsize=(10, 6))
    # bars = ax.bar(models, scores, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # for bar, score in zip(bars, scores):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
    #             f'{score:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # ax.set_ylabel('Final Model Score (%)')
    # ax.set_title('Q→AR: Final Model Score by Model', fontsize=16, fontweight='bold')
    # ax.set_ylim(0, max(scores) * 1.15)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # save(fig, '04_model_score_bar.png')


def plot_05_score_breakdown(df):
    """Grouped bar chart: Answer / Rationale / Model scores."""
    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    n = len(models)
    x = np.arange(n)
    width = 0.25

    ans_scores = df_s['Final Answer Score'].values
    rat_scores = df_s['Final Rationale Score'].values
    mod_scores = df_s['Final Model Score'].values

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width, ans_scores, width, label='Q→A (Answer)', color='#4E79A7', edgecolor='white')
    b2 = ax.bar(x,         rat_scores, width, label='Q→R (Rationale)', color='#59A14F', edgecolor='white')
    b3 = ax.bar(x + width, mod_scores, width, label='Q→AR (Combined)', color='#E15759', edgecolor='white')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score (%)')
    ax.set_title('Score Breakdown: Answer vs Rationale vs Combined', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(max(ans_scores), max(rat_scores), max(mod_scores)) * 1.15)

    ax.legend(frameon=True, fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '05_score_breakdown_bar.png')


def plot_06_scaling_law(df):
    """Line chart: Scaling law (parameter size vs scores)."""
    # Build (param_size, scores) pairs for models with known sizes
    records = []
    for _, row in df.iterrows():
        model = row['Model Name']
        if model in PARAM_SIZES:
            records.append({
                'model': model,
                'params': PARAM_SIZES[model],
                'Answer': row['Final Answer Score'],
                'Rationale': row['Final Rationale Score'],
                'Combined': row['Final Model Score'],
            })
    if not records:
        return

    rdf = pd.DataFrame(records).sort_values('params')
    params = rdf['params'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    line_styles = [
        ('Answer',    'Q→A (Answer)',    '#4E79A7', 'o'),
        ('Rationale', 'Q→R (Rationale)', '#59A14F', 's'),
        ('Combined',  'Q→AR (Combined)', '#E15759', 'D'),
    ]
    for col, label, color, marker in line_styles:
        vals = rdf[col].values
        ax.plot(params, vals, marker=marker, label=label, color=color,
                linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        for p, v, m in zip(params, vals, rdf['model']):
            ax.annotate(f'{v:.1f}', (p, v), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Parameter Size (Billions)')
    ax.xaxis.labelpad = 15

    ax.set_ylabel('Score (%)')
    ax.set_title('Scaling Law: Score vs Model Size', fontsize=16, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}B'))
    ax.set_xticks(params)

    # Add billion parameters labels on x-axis
    # for i, p in enumerate(params):
    #     ax.text(p, -2, f'{int(p)}B', ha='center', va='top', fontsize=9, fontweight='bold')
    ax.set_xticklabels([f'{int(p)}B' for p in params])

    # Add model name labels on x-axis
    # ax.set_xticklabels([short_name(m) for m in rdf['model']], fontsize=9)
    # Add some space as some words are overlapped
    # plt.xticks(rotation=25, ha='right')
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)

    ax.legend(frameon=True, fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '06_scaling_law_line.png')


# ─── SECTION C: DEEP ANALYSIS ───────────────────────────────────────────────
def plot_07_answer_vs_rationale(df):
    """Scatter plot: Answer score vs Rationale score gap with highlighted gap values."""
    fig, ax = plt.subplots(figsize=(9, 8))

    # Diagonal reference line
    ax.plot([0, 100], [0, 100], '--', color='#AAAAAA', linewidth=1, zorder=1, label='Balanced (A=R)')

    for _, row in df.iterrows():
        model = row['Model Name']
        ans = row['Final Answer Score']
        rat = row['Final Rationale Score']
        color = get_color(model)
        gap = rat - ans  # positive = rationale > answer

        # Draw a vertical dashed line from the diagonal (ans, ans) to the point (ans, rat)
        ax.plot([ans, ans], [ans, rat],
                color=color, linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)

        # Small horizontal tick at the diagonal end for clarity
        ax.plot([ans - 1, ans + 1], [ans, ans],
                color=color, linewidth=1.5, alpha=0.5, zorder=2)

        # Gap label at the midpoint of the vertical line
        mid_y = (ans + rat) / 2
        gap_sign = '+' if gap > 0 else ''
        ax.text(ans - 3, mid_y, f'{gap_sign}{gap:.1f}',
                ha='left', va='center', fontsize=8, fontweight='bold',
                color=color, alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.6, linewidth=0.8))

        # Scatter point
        ax.scatter(ans, rat,
                   color=color, s=200, zorder=3, edgecolors='white', linewidth=1.5)

        # Model name annotation
        ax.annotate(short_name(model),
                    (ans, rat),
                    textcoords='offset points', xytext=(-100, -5),
                    fontsize=10, fontweight='bold', color=color)

    ax.set_xlabel('Final Answer Score (%)')
    ax.set_ylabel('Final Rationale Score (%)')
    ax.set_title('Answer Score vs Rationale Score', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.legend(frameon=True, loc='lower right', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '07_answer_vs_rationale_scatter.png')

def _radar_chart(df, categories, score_col_fn, title, filename, cat_display_fn=None):
    """Generic radar chart helper."""
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor('white')

    df_s = sort_by_score(df)
    for _, row in df_s.iterrows():
        model = row['Model Name']
        values = [row[score_col_fn(c)] for c in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=short_name(model),
                color=get_color(model), markersize=6)
        ax.fill(angles, values, alpha=0.08, color=get_color(model))

    display_labels = [cat_display_fn(c) if cat_display_fn else c for c in categories]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='#666666')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=10)
    save(fig, filename)


def plot_08_domain_radar(df):
    _radar_chart(
        df, DOMAIN_LABELS,
        lambda d: f'{d} Score',
        'Domain Performance: Q→AR (Final Model Score)',
        '08_domain_radar.png',
        cat_display_fn=lambda d: DOMAIN_FULL.get(d, d)
    )


def plot_09_qtype_radar(df):
    _radar_chart(
        df, QTYPE_LABELS,
        lambda t: f'{t} Score',
        'Question Type Performance: Q→AR (Final Model Score)',
        '09_qtype_radar.png',
        cat_display_fn=lambda t: QTYPE_FULL.get(t, t)
    )


def plot_10_calc_split_bar(df):
    """Split bar chart: Calc vs Non-Calc."""
    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    n = len(models)
    x = np.arange(n)
    width = 0.35

    calc = df_s['Calc Score'].values
    noncalc = df_s['Non-Calc Score'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, calc,    width, label='Calculation', color='#4E79A7', edgecolor='white')
    b2 = ax.bar(x + width / 2, noncalc, width, label='Non-Calculation', color='#F28E2B', edgecolor='white')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score (%)')
    ax.set_title('Calc vs Non-Calc Performance: Q→AR', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(max(calc), max(noncalc)) * 1.18)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '10_calc_vs_noncalc_bar.png')


# ─── SECTION D: ERROR ANALYSIS ──────────────────────────────────────────────

def plot_11_error_heatmap(df):
    """Heatmap: Error code distribution (E0–E11)."""
    df_s = sort_by_score(df)
    error_codes = [f'E{i}' for i in range(12)]
    models = [short_name(m) for m in df_s['Model Name']]

    matrix = df_s[error_codes].values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(error_codes)))
    ax.set_xticklabels(error_codes, fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(error_codes)):
            val = int(matrix[i, j])
            text_color = 'white' if val > matrix.max() * 0.6 else '#222222'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=text_color)

    ax.set_title('Error Code Distribution (E0–E11) by Model', fontsize=16, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Frequency', fontsize=11)
    save(fig, '11_error_heatmap.png')

def plot_12_answer_correctness(df):
    """Stacked horizontal bar: Answer correctness %."""

    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    n = len(models)
    y = np.arange(n)

    incorrect = df_s['Ans Incorrect %'].values
    partial   = df_s['Ans Partial %'].values
    correct   = df_s['Ans Correct %'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.barh(y, incorrect, height=0.5, label='Incorrect', color='#E15759', edgecolor='white')
    b2 = ax.barh(y, partial,   height=0.5, left=incorrect, label='Partially Correct', color='#F28E2B', edgecolor='white')
    # b3 = ax.barh(y, correct,   height=0.5, left=incorrect + partial, label='Entirely Correct', color='#59A14F', edgecolor='white')

    # Label each segment
    for bars, vals in [(b1, incorrect), (b2, partial)]:
        for bar, val in zip(bars, vals):
            if val > 5:  # skip tiny segments
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    # for bars, vals in [(b1, incorrect), (b2, partial), (b3, correct)]:
    #     for bar, val in zip(bars, vals):
    #         if val > 5:  # skip tiny segments
    #             ax.text(bar.get_x() + bar.get_width() / 2,
    #                     bar.get_y() + bar.get_height() / 2,
    #                     f'{val:.1f}%', ha='center', va='center',
    #                     fontsize=9, fontweight='bold', color='white')

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Answer Correctness Distribution (FB & OE Questions)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 105)

    ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1, fontsize=16)
    # ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '12_answer_correctness_stacked.png')

def plot_13_rationale_correctness(df):
    """Stacked horizontal bar: Rationale correctness %."""
    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    n = len(models)
    y = np.arange(n)

    incorrect = df_s['Rat Incorrect %'].values
    partial   = df_s['Rat Partial %'].values
    correct   = df_s['Rat Correct %'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.barh(y, incorrect, height=0.5, label='Incorrect', color='#E15759', edgecolor='white')
    b2 = ax.barh(y, partial,   height=0.5, left=incorrect, label='Partially Correct', color='#F28E2B', edgecolor='white')
    # b3 = ax.barh(y, correct,   height=0.5, left=incorrect + partial, label='Entirely Correct', color='#59A14F', edgecolor='white')

    for bars, vals in [(b1, incorrect), (b2, partial)]:
        for bar, val in zip(bars, vals):
            # Fix if val < 5 the values are diaplayed properly without overlapping
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
            else:
                ax.text(bar.get_x() + bar.get_width() - 2.9,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', ha='left', va='center',
                        fontsize=7, fontweight='bold', color='white')

    # for bars, vals in [(b1, incorrect), (b2, partial), (b3, correct)]:
    #     for bar, val in zip(bars, vals):
    #         if val > 5:
    #             ax.text(bar.get_x() + bar.get_width() / 2,
    #                     bar.get_y() + bar.get_height() / 2,
    #                     f'{val:.1f}%', ha='center', va='center',
    #                     fontsize=9, fontweight='bold', color='white')

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Rationale Correctness Distribution (All Questions)', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 105)

    ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1, fontsize=16)

    # ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '13_rationale_correctness_stacked.png')


# ─── SECTION E: ADVANCED ────────────────────────────────────────────────────
def plot_14_avg_penalty(df):
    """Horizontal bar chart: Average penalty per model."""
    df_s = sort_by_score(df)
    models = [short_name(m) for m in df_s['Model Name']]
    n = len(models)
    y = np.arange(n)
    height = 0.3

    ans_penalty = 100 - df_s['Final Answer Score'].values
    rat_penalty = 100 - df_s['Final Rationale Score'].values
    avg_penalty = (ans_penalty + rat_penalty) / 2

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.barh(y - height, ans_penalty, height, label='Answer Penalty', color='#E15759', edgecolor='white')
    b2 = ax.barh(y,          rat_penalty, height, label='Rationale Penalty', color='#F28E2B', edgecolor='white')
    b3 = ax.barh(y + height, avg_penalty, height, label='Avg Penalty', color='#4E79A7', edgecolor='white')

    for bars in [b1, b2, b3]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{w:.1f}', ha='left', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel('Penalty')
    ax.set_title('Average Penalty per Model', fontsize=16, fontweight='bold')
    ax.set_xlim(0, max(ans_penalty.max(), rat_penalty.max(), avg_penalty.max()) * 1.15)
    ax.legend(frameon=True, loc='upper right', bbox_to_anchor=(1, 0.3), ncol=1, fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, '14_avg_penalty_bar.png')

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate visualisation graphs from model evaluation CSVs.')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='Paths to _result_by_model.csv files')
    args = parser.parse_args()

    # Load and merge all CSVs
    dfs = []
    for f in args.input_files:
        if not os.path.exists(f):
            print(f"❌ File not found: {f}")
            return
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    print(f"\n📊 Loaded {len(df)} model(s): {', '.join(df['Model Name'].tolist())}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    print("\n--- Section A: Dataset Construction ---")
    plot_01_question_types_pie(df)
    plot_02_domains_pie(df)
    plot_03_calc_pie(df)

    print("\n--- Section B: Core Performance ---")
    plot_04_model_score_bar(df)
    plot_05_score_breakdown(df)
    plot_06_scaling_law(df)

    print("\n--- Section C: Deep Analysis ---")
    plot_07_answer_vs_rationale(df)
    plot_08_domain_radar(df)
    plot_09_qtype_radar(df)
    plot_10_calc_split_bar(df)

    print("\n--- Section D: Error Analysis ---")
    plot_11_error_heatmap(df)
    plot_12_answer_correctness(df)
    plot_13_rationale_correctness(df)

    print("\n--- Section E: Advanced ---")
    plot_14_avg_penalty(df)

    print(f"\n✅ All 14 graphs saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
