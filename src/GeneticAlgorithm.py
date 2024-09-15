import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

class GeneticAlgorithm:
    def __init__(self, model_class, train_dataset, val_dataset, vocab_size, output_dim, device, pop_size=10, generations=10, mutation_rate=0.2, elite_fraction=0.1, early_stopping_threshold=0.01):
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.device = device
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.early_stopping_threshold = early_stopping_threshold
        self.fitness_history = []

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            params = {
                'embedding_dim': random.choice([50, 100, 150]),
                'hidden_dim': random.choice([64, 128, 256]),
                'n_layers': random.choice([1, 2, 3]),
                'bidirectional': random.choice([True, False]),
                'dropout': random.uniform(0.1, 0.5),
                'learning_rate': random.uniform(0.0001, 0.01),
                'batch_size': random.choice([16, 32, 64, 128])
            }
            population.append(params)
        return population

    def fitness(self, model_params):
        # Create model with current parameters
        model = self.model_class(
            self.vocab_size,
            model_params['embedding_dim'],
            model_params['hidden_dim'],
            self.output_dim,
            model_params['n_layers'],
            model_params['bidirectional'],
            model_params['dropout']
        )
        model = model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=model_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Create DataLoader based on the selected batch size
        train_loader = DataLoader(self.train_dataset, batch_size=model_params['batch_size'], shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=model_params['batch_size'], shuffle=False)

        # Train the model and evaluate its accuracy on the validation set
        train_loss, train_accuracy = self.train_model(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = self.evaluate_model(model, val_loader, criterion)

        return val_accuracy

    def train_model(self, model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / len(train_loader.dataset)
        return epoch_loss, epoch_accuracy

    def evaluate_model(self, model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = correct_predictions / len(val_loader.dataset)
        return epoch_loss, epoch_accuracy

    def elitism(self, population, fitness_scores):
        combined = list(zip(fitness_scores, population))
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        sorted_population = [x[1] for x in combined_sorted]
        elite_size = int(len(population) * self.elite_fraction)
        return sorted_population[:elite_size]

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            participants = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = max(participants, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()
        for param in parent1.keys():
            if random.random() < 0.5:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        return child1, child2

    def mutate(self, individual, generation):
        mutation_rate = self.adaptive_mutation_rate(generation)
        for param in individual.keys():
            if random.random() < mutation_rate:
                if param == 'embedding_dim':
                    individual[param] = random.choice([50, 100, 150])
                elif param == 'hidden_dim':
                    individual[param] = random.choice([64, 128, 256])
                elif param == 'n_layers':
                    individual[param] = random.choice([1, 2, 3])
                elif param == 'bidirectional':
                    individual[param] = random.choice([True, False])
                elif param == 'dropout':
                    individual[param] = random.uniform(0.1, 0.5)
                elif param == 'learning_rate':
                    individual[param] = random.uniform(0.0001, 0.01)
                elif param == 'batch_size':
                    individual[param] = random.choice([16, 32, 64, 128])
        return individual

    def adaptive_mutation_rate(self, generation):
        initial_rate = 0.2
        final_rate = 0.05
        return initial_rate - (initial_rate - final_rate) * (generation / self.generations)

    def early_stopping(self):
        if len(self.fitness_history) > 5:
            recent_improvements = [self.fitness_history[i + 1] - self.fitness_history[i] for i in range(-5, -1)]
            if all(improvement < self.early_stopping_threshold for improvement in recent_improvements):
                return True
        return False

    def run(self):
        try:
            population = self.initialize_population()
            best_individual = None
            best_fitness = -float('inf')

            for generation in range(self.generations):
                fitness_scores = [self.fitness(individual) for individual in population]
                self.fitness_history.append(max(fitness_scores))

                if self.early_stopping():
                    print(f"Early stopping at generation {generation + 1}")
                    break

                elite_individuals = self.elitism(population, fitness_scores)
                selected_population = self.tournament_selection(population, fitness_scores)

                next_population = elite_individuals.copy()
                while len(next_population) < self.pop_size:
                    parent1, parent2 = random.sample(selected_population, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    next_population.append(self.mutate(child1, generation))
                    if len(next_population) < self.pop_size:
                        next_population.append(self.mutate(child2, generation))

                population = next_population

                current_best_individual = max(population, key=self.fitness)
                current_best_fitness = self.fitness(current_best_individual)
                if current_best_fitness > best_fitness:
                    best_individual = current_best_individual
                    best_fitness = current_best_fitness

                print(f'Generation {generation + 1}: Best Validation Accuracy = {best_fitness:.4f}')

            return best_individual
        except Exception as e:
            print(f'An error occurred: {e}')
            raise

# Genetic algorithm parameters
pop_size = 10
generations = 10
mutation_rate = 0.2
elite_fraction = 0.1
early_stopping_threshold = 0.01