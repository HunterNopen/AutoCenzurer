import sys
sys.path.append('./')

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from evaluation.metrics import (
    get_metrics_for_binary, get_metrics_for_multiclass, generate_confusion_matrix, Approach, Metric
)

BINARY_APPROACHES: list[Approach] = [
    Approach(key='qwen', file_path=f'artifacts/inference_results/qwen_binary.csv', display_name='Qwen-Instruct-3B fine-tuned'),
    Approach(key='llama', file_path=f'artifacts/inference_results/groq_binary.csv', display_name='llama-3.1-8b-instant'),
]

MULTICLASS_APPROACHES: list[Approach] = [
    Approach(key='qwen-mc', file_path=f'artifacts/inference_results/qwen_multiclass.csv', display_name='Qwen-Instruct-3B fine-tuned'),
    Approach(key='llama-mc', file_path=f'artifacts/inference_results/groq_multiclass.csv', display_name='llama-3.1-8b-instant'),
]

binary_metrics = get_metrics_for_binary(BINARY_APPROACHES)
multiclass_metrics = get_metrics_for_multiclass(MULTICLASS_APPROACHES)
binary_confusion_matrices = [generate_confusion_matrix(approach) for approach in BINARY_APPROACHES]
multiclass_confusion_matrices = [generate_confusion_matrix(approach) for approach in MULTICLASS_APPROACHES]

def expand_per_class_metrics(metrics: list[Metric]) -> list[Metric]:
    """Expand per-class metrics (arrays) into separate metric objects for each class."""
    expanded = []
    
    for metric in metrics:
        # Check if any value is an array (per-class metric)
        has_arrays = any(isinstance(v, np.ndarray) for v in metric.values.values())
        
        if has_arrays:
            # Get number of classes from first array value
            first_array = next(v for v in metric.values.values() if isinstance(v, np.ndarray))
            n_classes = len(first_array)
            
            # Create a metric for each class
            for class_idx in range(n_classes):
                class_metric = Metric(f"{metric.name}_class_{class_idx}")
                for approach, value in metric.values.items():
                    if isinstance(value, np.ndarray):
                        class_metric.values[approach] = float(value[class_idx])
                    else:
                        class_metric.values[approach] = float(value)
                expanded.append(class_metric)
        else:
            # Scalar metric, add as is
            expanded.append(metric)
    
    return expanded


def metric_to_bar_chart(metric: Metric) -> Figure:
    """Convert a Metric object to a bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    approaches = [app.display_name for app in metric.values.keys()]
    # All values should be scalar at this point
    values = [float(v) for v in metric.values.values()]
    
    ax.bar(approaches, values, color='steelblue', alpha=0.8)
    ax.set_ylabel(metric.name)
    ax.set_title(f'{metric.name} Comparison')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def metric_to_dataframe(metric: Metric) -> pd.DataFrame:
    """Convert a Metric object to a DataFrame."""
    # All values should be scalar at this point
    values = [f"{float(v):.4f}" for v in metric.values.values()]
    
    data = {
        'Approach': [app.display_name for app in metric.values.keys()],
        'Value': values
    }
    return pd.DataFrame(data)


def confusion_matrix_display_to_figure(display) -> Figure:
    """Convert ConfusionMatrixDisplay to a matplotlib Figure."""
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax)
    return fig


def create_ui():
    """Create the metrics comparison Gradio interface."""
    
    # State variables for managing UI
    state = {
        'approach_type': 'binary',
        'metric_idx': 0,
        'cm_idx': 0,
        'current_metrics': binary_metrics
    }
    
    with gr.Blocks(title="Metrics Comparison") as interface:
        gr.Markdown("# Metrics Comparison Dashboard")
        
        # ====== Approach Type Selector ======
        with gr.Row():
            approach_radio = gr.Radio(
                choices=['binary', 'multiclass'],
                value='binary',
                label='Select Approach Type'
            )
        
        # ====== Confusion Matrix Section ======
        gr.Markdown("## Confusion Matrix")
        with gr.Row():
            cm_plot = gr.Plot(label="Confusion Matrix")
            with gr.Column():
                cm_dropdown = gr.Dropdown(
                    choices=[app.display_name for app in BINARY_APPROACHES],
                    value=BINARY_APPROACHES[0].display_name,
                    label="Select Approach"
                )
        
        # ====== Metrics Section ======
        gr.Markdown("## Metrics Analysis")
        with gr.Row():
            metric_dropdown = gr.Dropdown(
                choices=['f1_score', 'accuracy', 'precision', 'recall'],
                value='f1_score',
                label='Select Metric'
            )
        
        with gr.Row():
            with gr.Column():
                metric_plot = gr.Plot(label="Metric Comparison")
            with gr.Column():
                metric_table = gr.DataFrame(label="Metric Values")
        
        # ====== Event Handlers ======
        
        def on_approach_change(approach_type_val):
            """Handle approach type change."""
            state['approach_type'] = approach_type_val
            state['cm_idx'] = 0
            state['metric_idx'] = 0
            
            approaches = BINARY_APPROACHES if approach_type_val == 'binary' else MULTICLASS_APPROACHES
            raw_metrics = binary_metrics if approach_type_val == 'binary' else multiclass_metrics
            
            # Expand per-class metrics for multiclass
            if approach_type_val == 'multiclass':
                metrics = expand_per_class_metrics(raw_metrics)
            else:
                metrics = raw_metrics
            
            approach_names = [app.display_name for app in approaches]
            metric_names = [m.name for m in metrics]
            
            # Update confusion matrix
            cm_displays = binary_confusion_matrices if approach_type_val == 'binary' else multiclass_confusion_matrices
            cm_fig = confusion_matrix_display_to_figure(cm_displays[0])
            
            # Update metric dropdown and display
            metric = metrics[0]
            metric_fig = metric_to_bar_chart(metric)
            metric_df = metric_to_dataframe(metric)
            
            # Store expanded metrics in state for use in other handlers
            state['current_metrics'] = metrics
            
            return (
                gr.Dropdown(choices=approach_names, value=approach_names[0]),
                cm_fig,
                gr.Dropdown(choices=metric_names, value=metric_names[0]),
                metric_fig,
                metric_df
            )
        
        def on_cm_dropdown_change(approach_name):
            """Handle confusion matrix dropdown change."""
            approaches = BINARY_APPROACHES if state['approach_type'] == 'binary' else MULTICLASS_APPROACHES
            cm_displays = binary_confusion_matrices if state['approach_type'] == 'binary' else multiclass_confusion_matrices
            
            # Find the index of the selected approach
            cm_idx = next(i for i, app in enumerate(approaches) if app.display_name == approach_name)
            state['cm_idx'] = cm_idx
            cm_fig = confusion_matrix_display_to_figure(cm_displays[cm_idx])
            
            return cm_fig
        
        def on_metric_change(metric_name):
            """Handle metric dropdown change."""
            metrics = state.get('current_metrics', binary_metrics)
            metric = next(m for m in metrics if m.name == metric_name)
            
            metric_fig = metric_to_bar_chart(metric)
            metric_df = metric_to_dataframe(metric)
            
            return metric_fig, metric_df
        
        # Wire events
        approach_radio.change(
            on_approach_change,
            inputs=[approach_radio],
            outputs=[cm_dropdown, cm_plot, metric_dropdown, metric_plot, metric_table]
        )
        
        cm_dropdown.change(
            on_cm_dropdown_change,
            inputs=[cm_dropdown],
            outputs=[cm_plot]
        )
        
        metric_dropdown.change(
            on_metric_change,
            inputs=[metric_dropdown],
            outputs=[metric_plot, metric_table]
        )
        
        # Initialize with binary metrics
        on_approach_load = gr.on(
            triggers=[interface.load],
            fn=lambda: on_approach_change('binary'),
            outputs=[cm_dropdown, cm_plot, metric_dropdown, metric_plot, metric_table]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_ui()
    interface.launch()
    
