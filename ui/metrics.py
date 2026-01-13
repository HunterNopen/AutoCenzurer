import sys
sys.path.append('./')

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure
import numpy as np

from evaluation.metrics import (
    APPROACHES, calculate_metrics, load_all_approaches, 
    generate_confusion_matrix
)
from evaluation.pipeline import Results

ENABLED = True

def create_confusion_matrix_plot(results: Results) -> Figure:
    """Create a confusion matrix visualization."""
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    display = generate_confusion_matrix(results.y_true, results.y_pred)
    display.plot(ax=ax)
    fig.tight_layout()
    return fig


def create_single_metric_plot(
    metric_name: str,
    all_results: dict[str, Results],
    enabled_approaches: dict[str, bool]
) -> Figure:
    """Create a bar chart for a single metric across enabled approaches."""
    # Filter enabled approaches
    enabled_keys = [key for key, enabled in enabled_approaches.items() if enabled]

    # Prepare labels and values
    approach_map = {app.key: app.display_name for app in APPROACHES}
    labels: list[str] = []
    values: list[float] = []

    for key in enabled_keys:
        if key in all_results:
            results = all_results[key]
            metrics = calculate_metrics(results.y_true, results.y_pred)
            if metric_name == 'Precision':
                values.append(metrics.precision)
            elif metric_name == 'Recall':
                values.append(metrics.recall)
            elif metric_name == 'Accuracy':
                values.append(metrics.accuracy)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
            labels.append(approach_map[key])

    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    if not labels:
        ax.text(0.5, 0.5, "No approaches enabled", ha='center', va='center')
        return fig

    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6)

    ax.set_xlabel('Approach')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    return fig


def create_metric_table(
    metric_name: str,
    all_results: dict[str, Results],
    enabled_approaches: dict[str, bool]
) -> pd.DataFrame:
    """Return a DataFrame with columns ['Approach', 'Metric Value'] for the metric."""
    enabled_keys = [key for key, enabled in enabled_approaches.items() if enabled]
    approach_map = {app.key: app.display_name for app in APPROACHES}
    rows: list[list[float | str]] = []
    for key in enabled_keys:
        if key in all_results:
            results = all_results[key]
            metrics = calculate_metrics(results.y_true, results.y_pred)
            if metric_name == 'Precision':
                value = metrics.precision
            elif metric_name == 'Recall':
                value = metrics.recall
            elif metric_name == 'Accuracy':
                value = metrics.accuracy
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
            rows.append([approach_map[key], round(value, 4)])
    return pd.DataFrame(rows, columns=["Approach", "Metric Value"])


def create_ui() -> gr.Blocks:
    """Create the Gradio UI for metrics comparison."""
    # Load all results
    all_results = load_all_approaches()
    
    # Initialize approach state
    enabled_approaches: dict[str, bool] = {app.key: ENABLED for app in APPROACHES}
    confusion_matrix_current_approach_idx = [0]
    
    def update_enabled_state(*toggle_values: bool) -> dict[str, bool]:
        """Update which approaches are enabled based on toggle states."""
        for i, toggle_value in enumerate(toggle_values):
            enabled_approaches[APPROACHES[i].key] = toggle_value
        
        return enabled_approaches
    
    def get_enabled_keys() -> list[str]:
        """Get list of currently enabled approach keys."""
        return [key for key in enabled_approaches.keys() if enabled_approaches[key]]
    
    def show_previous_confusion_matrix() -> tuple[Figure, str]:
        """Show confusion matrix for the previous enabled approach."""
        enabled = get_enabled_keys()
        if not enabled:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No approaches enabled", ha='center', va='center')
            fig.tight_layout()
            return fig, "No approaches enabled"
        
        if len(enabled) > 0:
            confusion_matrix_current_approach_idx[0] = (confusion_matrix_current_approach_idx[0] - 1) % len(enabled)
        
        key = enabled[confusion_matrix_current_approach_idx[0]]
        approach_name = next(app.display_name for app in APPROACHES if app.key == key)
        
        if key in all_results:
            fig = create_confusion_matrix_plot(all_results[key])
        else:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"No results for {approach_name}", ha='center', va='center')
            fig.tight_layout()
        
        return fig, approach_name
    
    def show_next_confusion_matrix() -> tuple[Figure, str]:
        """Show confusion matrix for the next enabled approach."""
        enabled = get_enabled_keys()
        if not enabled:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No approaches enabled", ha='center', va='center')
            fig.tight_layout()
            return fig, "No approaches enabled"
        
        if len(enabled) > 0:
            confusion_matrix_current_approach_idx[0] = (confusion_matrix_current_approach_idx[0] + 1) % len(enabled)
        
        key = enabled[confusion_matrix_current_approach_idx[0]]
        approach_name = next(app.display_name for app in APPROACHES if app.key == key)
        
        if key in all_results:
            fig = create_confusion_matrix_plot(all_results[key])
        else:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"No results for {approach_name}", ha='center', va='center')
            fig.tight_layout()
        
        return fig, approach_name
    
    def on_toggle_change(*toggle_values: bool) -> tuple[Figure, str, Figure, Figure, Figure, list[list], list[list], list[list]]:
        """Handle toggle changes and update all visualizations."""
        update_enabled_state(*toggle_values)
        confusion_matrix_current_approach_idx[0] = 0  # Reset to first approach
        
        # Update confusion matrix
        enabled = get_enabled_keys()
        if enabled and enabled[0] in all_results:
            cm_fig = create_confusion_matrix_plot(all_results[enabled[0]])
            approach_name = next(app.display_name for app in APPROACHES if app.key == enabled[0])
        else:
            cm_fig = Figure(figsize=(6, 4))
            ax = cm_fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No approaches enabled", ha='center', va='center')
            cm_fig.tight_layout()
            approach_name = "None"
        
        # Update separate metric plots
        precision_fig = create_single_metric_plot('Precision', all_results, enabled_approaches)
        recall_fig = create_single_metric_plot('Recall', all_results, enabled_approaches)
        accuracy_fig = create_single_metric_plot('Accuracy', all_results, enabled_approaches)
        
        precision_table = create_metric_table('Precision', all_results, enabled_approaches)
        recall_table = create_metric_table('Recall', all_results, enabled_approaches)
        accuracy_table = create_metric_table('Accuracy', all_results, enabled_approaches)
        
        return cm_fig, approach_name, precision_fig, recall_fig, accuracy_fig, precision_table, recall_table, accuracy_table
    
    with gr.Blocks(title="Classification Metrics Comparison") as demo:
        gr.Markdown("# Classification Metrics Comparison")
        
        # Toggle section
        with gr.Group():
            gr.Markdown("### Select Approaches to Display")
            toggle_inputs = []
            for approach in APPROACHES:
                toggle = gr.Checkbox(
                    label=approach.display_name,
                    value=ENABLED,
                    interactive=True
                )
                toggle_inputs.append(toggle)
        
        # Confusion Matrix section
        with gr.Group():
            gr.Markdown("### Confusion Matrix")
            with gr.Row():
                with gr.Column():
                    prev_btn = gr.Button("← Previous")
                with gr.Column():
                    approach_name_display = gr.Textbox(
                        label="Current Approach",
                        interactive=False
                    )
                with gr.Column():
                    next_btn = gr.Button("Next →")
            
            confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
        
        # Metrics sections
        with gr.Group():
            gr.Markdown("### Precision")
            precision_plot = gr.Plot(label="Precision")
            precision_table = gr.Dataframe(headers=["Approach", "Metric Value"], interactive=False)
        with gr.Group():
            gr.Markdown("### Recall")
            recall_plot = gr.Plot(label="Recall")
            recall_table = gr.Dataframe(headers=["Approach", "Metric Value"], interactive=False)
        with gr.Group():
            gr.Markdown("### Accuracy")
            accuracy_plot = gr.Plot(label="Accuracy")
            accuracy_table = gr.Dataframe(headers=["Approach", "Metric Value"], interactive=False)
        
        # Initialize displays
        initial_enabled = get_enabled_keys()
        if initial_enabled and initial_enabled[0] in all_results:
            initial_cm = create_confusion_matrix_plot(all_results[initial_enabled[0]])
            initial_name = next(app.display_name for app in APPROACHES if app.key == initial_enabled[0])
        else:
            initial_cm = Figure(figsize=(6, 4))
            ax = initial_cm.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No approaches enabled", ha='center', va='center')
            initial_cm.tight_layout()
            initial_name = "None"
        
        precision_initial = create_single_metric_plot('Precision', all_results, enabled_approaches)
        recall_initial = create_single_metric_plot('Recall', all_results, enabled_approaches)
        accuracy_initial = create_single_metric_plot('Accuracy', all_results, enabled_approaches)

        precision_table_initial = create_metric_table('Precision', all_results, enabled_approaches)
        recall_table_initial = create_metric_table('Recall', all_results, enabled_approaches)
        accuracy_table_initial = create_metric_table('Accuracy', all_results, enabled_approaches)

        confusion_matrix_plot.value = initial_cm
        approach_name_display.value = initial_name
        precision_plot.value = precision_initial
        recall_plot.value = recall_initial
        accuracy_plot.value = accuracy_initial
        precision_table.value = precision_table_initial
        recall_table.value = recall_table_initial
        accuracy_table.value = accuracy_table_initial
        
        # Connect events
        for toggle in toggle_inputs:
            toggle.change(
                on_toggle_change,
                inputs=toggle_inputs,
                outputs=[
                    confusion_matrix_plot,
                    approach_name_display,
                    precision_plot,
                    recall_plot,
                    accuracy_plot,
                    precision_table,
                    recall_table,
                    accuracy_table,
                ]
            )
        
        prev_btn.click(
            show_previous_confusion_matrix,
            outputs=[confusion_matrix_plot, approach_name_display]
        )
        
        next_btn.click(
            show_next_confusion_matrix,
            outputs=[confusion_matrix_plot, approach_name_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
