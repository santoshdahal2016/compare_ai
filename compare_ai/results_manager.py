import json

class Result:
    def __init__(self, predictions):
        self.predictions = predictions

    def __str__(self):
        return f"Predictions: {self.predictions}"
    

    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.predictions, file)

    def load_from_file(self, file_path):
        with open(file_path, 'r') as file:
            self.predictions = json.load(file)
 