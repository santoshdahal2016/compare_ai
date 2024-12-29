def run_predictions(eval_dataset, models):
    """
    Run predictions on evaluation dataset using provided models.
    
    Args:
        eval_dataset: Dataset to evaluate models on
        models: List of models to compare
        
    Returns:
        dict: Dictionary containing prediction results for each model
    """

    results = {}
    
    for model in models:
        model_predictions = []
        
        for item in eval_dataset:
            try:
                prediction = model.predict(item)
                model_predictions.append({
                    'input': item,
                    'prediction': prediction
                })
            except Exception as e:
                print(f"Error running prediction for model {model}: {str(e)}")
                continue
                
        results[model.name] = model_predictions
    
    return results
