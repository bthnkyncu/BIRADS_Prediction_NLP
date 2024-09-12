import os
import re


class DataPreprocessor:
    def __init__(self, directory_paths):
        self.directory_paths = directory_paths
        self.reports = []
        self.labels = []
        self.missing_labels = []

    def load_data(self):
        for directory_path in self.directory_paths:
            birads_value = self._determine_birads_value(directory_path)
            if birads_value:
                self._process_directory(directory_path, birads_value)
        return self.reports, self.labels, self.missing_labels

    def _determine_birads_value(self, directory_path):
        if "Birads-1" in directory_path:
            return "1"
        elif "Birads-2" in directory_path:
            return "2"
        elif "Birads-3" in directory_path:
            return "3"
        elif "Birads-4" in directory_path:
            return "4"
        elif "Birads-5" in directory_path:
            return "5"
        return None

    def _process_directory(self, directory_path, birads_value):
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    sonuc_line = re.search(r'(SONUÇ|RADYOLOJİK TANI|Sonuc|SONUC)\s*[:\-]\s*B[İI]RADS[\s\-]*(\d+)', content, re.IGNORECASE)
                    if sonuc_line:
                        self.reports.append(content)
                        self.labels.append(sonuc_line.group(2))
                    else:
                        self.missing_labels.append(filename)