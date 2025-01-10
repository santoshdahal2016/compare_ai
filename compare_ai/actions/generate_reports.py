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
            
            if not isinstance(labels, (int, float, dict)):
                raise ValueError("Each value in 'models' must be an int, float, or dictionary.")
            
            # If labels is a dictionary, validate the label keys and values
            if isinstance(labels, dict):
                for label, value in labels.items():
                    if not isinstance(label, str):
                        raise ValueError("Each key in the labels dictionary must be a string.")
                    
                    if not isinstance(value, (int, float)):
                        raise ValueError("Each value in the labels dictionary must be an int or float.")
    
    return True

def generate_comparison_plot(data):
    """
    Generates a colorful bar chart for model comparison based on the data provided.
    """
    models = []
    accuracy_values = []

    # Collect model data for categorical types
    for item in data:
        if item["type"] == "categorical":
            for model_name, labels in item["models"].items():
                models.append(model_name)
                accuracy = sum(labels.values())  # Sum up the scores for each model
                accuracy_values.append(accuracy)

    # Plotting the data with colors
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Paired(np.linspace(0, 1, len(models)))  # Generate a color palette
    plt.bar(models, accuracy_values, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.xticks(rotation=45, ha='right')

    # Save the plot to a temporary file
    plot_file = "model_comparison_plot.png"
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    return plot_file

def generate_likert_scale_plot(data):
    """
    Generates a Likert Scale bar chart for categorical metrics based on the data provided.
    """
    # Prepare data for Likert scale comparison
    likert_categories = ['Inaccurate', 'Partially Accurate', 'Accurate']
    models = list(set([model for item in data for model in item["models"].keys()]))
    likert_data = {model: [0, 0, 0] for model in models}

    # Fill in the Likert scale values
    for item in data:
        if item["type"] == "categorical":
            for model_name, labels in item["models"].items():
                for label, count in labels.items():
                    if label == 'inaccurate':
                        likert_data[model_name][0] += count
                    elif label == 'partially_accurate':
                        likert_data[model_name][1] += count
                    elif label == 'accurate':
                        likert_data[model_name][2] += count

    # Plotting the Likert scale data
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(models))

    # Plot each Likert scale category
    for i, category in enumerate(likert_categories):
        category_values = [likert_data[model][i] for model in models]
        ax.bar(index + i * bar_width, category_values, bar_width, label=category)

    ax.set_xlabel('Models')
    ax.set_ylabel('Count')
    ax.set_title(' Comparison of Models')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.legend()

    # Save the plot to a temporary file
    likert_plot_file = "likert_scale_comparison.png"
    plt.tight_layout()
    plt.savefig(likert_plot_file)
    plt.close()

    return likert_plot_file

def generate_pdf_report(data, plot_file, likert_plot_file):
    """
    Generates a PDF report with the model comparison plot and Likert scale comparison.
    """
    file_name = "model_comparison_report.pdf"
    
    c = canvas.Canvas(file_name, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Add a title
    c.drawString(100, 750, "Model Comparison Report")
    
    # Insert the model comparison plot image
    c.drawImage(plot_file, 100, 450, width=400, height=300)

    # Insert the Likert scale comparison plot
    c.drawImage(likert_plot_file, 100, 100, width=400, height=300)

    # Save the PDF
    c.save()

    # Clean up the plot files
    os.remove(plot_file)
    os.remove(likert_plot_file)

    return file_name

def generate_report(data):
    """Generate a report from the validated data."""
    validate_data_format(data)
    
    # Generate the comparison plot
    plot_file = generate_comparison_plot(data)
    
    # Generate the Likert scale plot
    likert_plot_file = generate_likert_scale_plot(data)
    
    # Generate the PDF report
    pdf_file = generate_pdf_report(data, plot_file, likert_plot_file)
    return pdf_file


# Example usage with dynamically loaded data
if __name__ == "__main__":
    # Example data
    data = [
        {
            "name": "usability_gap",
            "type": "categorical",  
            "models":{
                "gpt-3": {
                    "partially_accurate": 4,
                    "inaccurate": 2,
                    "accurate": 1
                },
                "gpt-4": {
                    "partially_accurate": 2,
                    "inaccurate": 1,
                    "accurate": 4
                }
            }
        },
        {
            "name": "BLUE Score",
            "type": "numerical",  
            "models":{
                "gpt-3": 98,
                "gpt-4": 99
            }
        }
    ]

    try:
        report = generate_report(data)
        print(f"Report generated successfully: {report}")
    except ValueError as e:
        print(f"Error: {e}")
