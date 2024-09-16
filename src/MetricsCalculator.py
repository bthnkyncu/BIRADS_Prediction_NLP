import torch
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricsCalculator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def calculate_metrics(self):
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for texts, labels in self.data_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate macro and micro averages for precision, recall, and F1-score
        precision_macro = precision_score(all_labels, all_predictions, average='macro')
        recall_macro = recall_score(all_labels, all_predictions, average='macro')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        
        precision_micro = precision_score(all_labels, all_predictions, average='micro')
        recall_micro = recall_score(all_labels, all_predictions, average='micro')
        f1_micro = f1_score(all_labels, all_predictions, average='micro')

        return precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

    def print_metrics(self):
        precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro = self.calculate_metrics()
        print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1-score: {f1_macro:.4f}")
        print(f"Micro Average - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1-score: {f1_micro:.4f}")