import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import os

def validate_data_format(data):
    """
    Validates the structure of the given data format.
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list.")

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item in data must be a dictionary.")
        
        # Validate "name"
        if "name" not in item or not isinstance(item["name"], str):
            raise ValueError("Each item must have a 'name' key with a string value.")
        
        # Validate "type"
        if "type" not in item or not isinstance(item["type"], str):
            raise ValueError("Each item must have a 'type' key with a string value.")
        
        if item["type"] not in ["categorical", "numerical"]:
            raise ValueError("The 'type' must be either 'categorical' or 'numerical'.")
        
        # Validate "models"
        if "models" not in item or not isinstance(item["models"], dict):
            raise ValueError("Each item must have a 'models' key with a dictionary value.")
        
        # Validate the "models" dictionary
        for model_name, labels in item["models"].items():
            if not isinstance(model_name, str):
                raise ValueError("Each key in 'models' must be a string representing a model name.")
            
            if item["type"] == "numerical" and not isinstance(labels, (int, float)):
                raise ValueError("For numerical metrics, values must be int or float.")
            
            if item["type"] == "categorical" and not isinstance(labels, dict):
                raise ValueError("For categorical metrics, values must be a dictionary.")
            
            if item["type"] == "categorical":
                for label, value in labels.items():
                    if not isinstance(label, str):
                        raise ValueError("Each key in the labels dictionary must be a string.")
                    
                    if not isinstance(value, (int, float)):
                        raise ValueError("Each value in the labels dictionary must be an int or float.")
    
    return True

def generate_metric_plot(metric_name, metric_type, models):
    """
    Generates a plot for a single metric based on its type and model data.
    """
    plot_file = f"{metric_name.replace(' ', '_')}_plot.png"
    
    if metric_type == "numerical":
        # Bar chart for numerical metrics
        model_names = list(models.keys())
        values = list(models.values())
        colors = plt.cm.Paired(np.linspace(0, 1, len(model_names)))
        
        plt.figure(figsize=(8, 5))
        plt.bar(model_names, values, color=colors)
        plt.xlabel("Models")
        plt.ylabel("Value")
        # plt.title(f"Numerical Metric: {metric_name}")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
    
    elif metric_type == "categorical":
        # Stacked bar chart for categorical metrics
        labels = set(label for model in models.values() for label in model.keys())
        label_indices = {label: i for i, label in enumerate(labels)}
        
        model_names = list(models.keys())
        data = np.zeros((len(model_names), len(labels)))
        
        for i, model in enumerate(model_names):
            for label, count in models[model].items():
                data[i, label_indices[label]] = count
        
        indices = np.arange(len(model_names))
        bar_width = 0.8 / len(labels)
        
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.bar(indices + i * bar_width, data[:, i], bar_width, label=label)
        
        plt.xlabel("Models")
        plt.ylabel("Count")
        # plt.title(f"Categorical Metric: {metric_name}")
        plt.xticks(indices + bar_width * (len(labels) / 2), model_names, rotation=45, ha="right")
        plt.legend(title="Categories")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
    
    return plot_file

def generate_pdf_report(data, plot_files):
    """
    Generates a PDF report with plots for each metric.
    """
    file_name = "model_comparison_report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Model Comparison Report")
    
    y_position = 700
    for metric_name, plot_file in plot_files:
        c.drawString(50, y_position, metric_name)
        c.drawImage(plot_file, 50, y_position - 310, width=500, height=300)
        y_position -= 350
        if y_position < 100:
            c.showPage()
            y_position = 750

    c.save()

    # Clean up plot files
    for _, plot_file in plot_files:
        os.remove(plot_file)

    return file_name

def generate_report(data):
    """Generate a detailed report with individual metric plots."""
    validate_data_format(data)
    
    # Generate plots for each metric
    plot_files = []
    for item in data:
        plot_file = generate_metric_plot(item["name"], item["type"], item["models"])
        plot_files.append((item["name"], plot_file))
    
    # Generate the PDF report
    pdf_file = generate_pdf_report(data, plot_files)
    return pdf_file
