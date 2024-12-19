class Result:
    def __init__(self, predictions):
        self.predictions = predictions

    def __str__(self):
        return f"Predictions: {self.predictions}"