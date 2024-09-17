import torch
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Main:
    def __init__(self, directory_paths):
        self.preprocessor = DataPreprocessor(directory_paths)

    def execute(self):
        # Load data
        reports, labels, missing_labels = self.preprocessor.load_data()

        # Text preprocessing, tokenization, and splitting data
        MAX_NB_WORDS = 50000
        MAX_SEQUENCE_LENGTH = 250
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(reports)
        sequences = tokenizer.texts_to_sequences(reports)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

        # Instantiate dataset objects and loaders
        train_dataset = BIRADSDataset(X_train, Y_train)
        test_dataset = BIRADSDataset(X_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Run Genetic Algorithm
        ga = GeneticAlgorithm(10, 10, 0.2, 0.1, 0.01)
        best_params = ga.run()
        print(f'Best Parameters: {best_params}')

        # Instantiate and train the model with best parameters
        best_model = X_LSTM(
            vocab_size=len(tokenizer.word_index) + 1,
            embedding_dim=best_params['embedding_dim'],
            hidden_dim=best_params['hidden_dim'],
            output_dim=6,  # BIRADS classification
            n_layers=best_params['n_layers'],
            bidirectional=best_params['bidirectional'],
            dropout=best_params['dropout']
        )
        best_model = best_model.to(device)
        optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate the model
        trainer = Trainer(best_model, train_loader, test_loader, device, optimizer, criterion)
        trainer.train_model()
        trainer.evaluate_model()

        # Calculate and display metrics
        metrics_calculator = MetricsCalculator(best_model, test_loader, device)
        metrics_calculator.print_metrics()