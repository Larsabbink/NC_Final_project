import copy
import sys

from game import Game
from model import Model
from random_player import RandomPlayer

import numpy as np
import random 

N = 50 # Population size
K = 6 # Amount of matches each network plays each generation

class Tournament:
    def __init__(self):
        self.population: list[Model] = self.create_population()
        self.scores = []

    def create_population(self):
        population = []

        for _ in range(N):
            population.append(Model())

        return population
    
    def play_tournament_round(self):
        scores = dict.fromkeys(list(range(0, N)), 0)
        
        for _ in range(K):
            schedule = iter(np.random.permutation(N))
            matches = zip(schedule, schedule)
            for player_1, player_2 in matches:
                game = Game(self.population[player_1], self.population[player_2])
                score_player_1, score_player_2 = game.play()

                scores[player_1] += score_player_1
                scores[player_2] += score_player_2
                
            
        # sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # sorted_models = list(zip(*sorted_scores))[0]
        
        return scores
    
    def mutate(self, sorted_models, p, k, keep):
        new_population = []

        for _ in range(N - keep):
            selection = random.sample(range(0, N), k)

            best_index = 50
            best_model = 0

            for i in selection:
                index = sorted_models.index(i)
                if index < best_index:
                    best_model = i
                    best_index = index

            model = copy.deepcopy(self.population[best_model])
            model.mutate(p)
            new_population.append(model)

        for i in range(keep):
            new_population.append(self.population[sorted_models[i]])

        
        self.population = new_population

    def evaluate_best_model(self, best_player):
        random_player = RandomPlayer()
        total_score_best_player = 0
        
        for _ in range(10):
            game = Game(best_player, random_player)
            score_best_player, _ = game.play()
            total_score_best_player += score_best_player

        self.scores.append(total_score_best_player)

    def main_loop(self, p, k, keep):
        rounds = 30

        for i in range(rounds):
            scores = self.play_tournament_round()

            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            sorted_models = list(zip(*sorted_scores))[0]

            self.evaluate_best_model(self.population[sorted_models[0]])
            self.mutate(sorted_models, p, k, keep)
            print(f'Iteration: {i + 1}, score: {self.scores[i]}')


def main():

    p = float(sys.argv[1])

    # ps = [0.01, 0.05, 0.1, 0.2] -- Let user enter p via command line such that 4 threads can be run at the same time
    ks = [2, 4, 6, 8]
    keeps = [0, 5, 10]

    for k in ks:
        for keep in keeps:
            tournament = Tournament()
            tournament.main_loop(p, k, keep)
            
            file = open(f'p_{p}-k_{k}-keep_{keep}.txt', 'w')

            file.write('\n'.join(str(score) for score in tournament.scores))

    test = [1,2,3,4,5]

    for item in test:
        file = open("test.txt", 'w')
        file.write('\n'.join(str(item) for item in test))

if __name__ == "__main__":
    main()