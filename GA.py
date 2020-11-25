# Genetic Algorithm
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
from EvaluationTool import EvaluationTool


class GA:
    def __init__(self, pop_size, max_gen=100, crossover_rate=0.6, mutation_rate=0.1):
        # initialization
        self.pop_size = pop_size  # Population
        self.max_gen = max_gen  # Maximum reproduction generation
        self.crossover_rate = crossover_rate  # Crossover rate
        self.mutation_rate = mutation_rate  # Mutation rate
        self.genes_num = None  # Number of genes
        self.clf_list = None
        self.target_data = None
        self.target_label = None
        self.pop_f1_list = []  # Record the f1 value of each individual in the population to avoid double counting

    def fit(self, source_data_list, source_label_list, target_data, target_label):
        # Get a bunch of basic classifiers
        clf_list = self.get_base_clf(source_data_list, source_label_list, target_data, target_label)
        self.clf_list = clf_list
        # Individual -> Chromosome -> list of genes
        # Determine the number of genes = Number of classifiers + Threshold
        genes_num = len(clf_list) + 1
        self.genes_num = genes_num
        # Initial population
        pop = np.random.random((self.pop_size, self.genes_num))
        # The value range of the threshold in the last column is adjusted from (0,1) to (0,genes_num-1)
        pop[:, -1] = pop[:, -1] * (genes_num - 1)
        # Calculate the optimal solution in the initial population
        pre_solution, pre_f1 = self.get_best_from_pop(pop)
        # Breed
        cur_gen = 0  # Current algebra is 0
        while cur_gen < self.max_gen:
            temp_pop = pop.copy()
            pop_new = self.ga_generation(temp_pop)
            cur_solution, cur_f1 = self.get_best_from_pop(pop_new)
            if cur_f1 > pre_f1:
                pre_f1 = cur_f1
                pre_solution = cur_solution
            cur_gen += 1
        print(pre_f1)
        return pre_solution, pre_f1

    # Train to get a lot of basic classifiers
    def get_base_clf(self, source_data_list, source_label_list, target_data, target_label):
        # Add a part of the samples in the target to the source as a training set
        sdl, sll, td, tl = self.transfer_target_to_source(source_data_list, source_label_list,
                                                          target_data, target_label)
        self.target_data, self.target_label = td, tl
        clf_list = []
        for index in range(len(source_data_list)):
            # Logistic regression is used in the paper, and the effect is not very good
            clf = LogisticRegression()
            # Change to C4.5
            # clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=30)
            clf.fit(sdl[index], sll[index])
            clf_list.append(clf)
        # Use target training to get (N+1)th clf
        clf = LogisticRegression()
        clf.fit(td, tl)
        clf_list.append(clf)
        return clf_list

    # Transfer some samples from target to source
    # 20180806 Change the for loop to a list comprehension
    @staticmethod
    def transfer_target_to_source(source_data_list, source_label_list, target_data, target_label):
        # Randomly remove 10% of samples
        t_sample_num = target_data.shape[0]
        t_remove_num = round(t_sample_num * 0.1)  # Round up
        t_remove_index = random.sample(range(0, t_sample_num), t_remove_num)
        t_other_index = [i for i in range(0, t_sample_num) if i not in t_remove_index]
        # New target
        target_data_new = np.array([target_data[i, :] for i in t_other_index])
        target_label_new = np.array([target_label[i] for i in t_other_index])
        # Samples and labels removed
        temp_data = np.array([target_data[i, :] for i in t_remove_index])
        temp_label = np.array([target_label[i] for i in t_remove_index])
        # New source
        temp_data_list = [np.append(source_data_list[i], temp_data, axis=0) for i in range(len(source_data_list))]
        temp_label_list = [np.append(source_label_list[i], temp_label, axis=0) for i in range(len(source_label_list))]
        return temp_data_list, temp_label_list, target_data_new, target_label_new

    # Select the best individual (chromosome) from the population
    def get_best_from_pop(self, pop):
        if len(self.pop_f1_list) != 0:
            self.pop_f1_list.clear()
        # Compare the F1 value of each individual
        pre_f1, best_index = 0, 0
        for index in range(pop.shape[0]):
            cur_f1 = self.score(pop[index])
            self.pop_f1_list.append(cur_f1)
            if cur_f1 > pre_f1:
                pre_f1 = cur_f1
                best_index = index
        return pop[best_index], pre_f1

    # Predict the label of the target data set and calculate the score
    def score(self, pop_item):
        # Calculate the score of each example in the target
        predict_label = []
        for sample in range(self.target_data.shape[0]):
            # comp = from i to N+1 (wight * clf prediction) / loc(sample)
            comp = 0
            for i in range(len(self.clf_list)):
                comp += pop_item[i] * self.clf_list[i].predict(self.target_data[sample].reshape(1, -1))[0]
            # comp /= self.target_data[sample, 0]
            if comp >= pop_item[-1]:
                predict_label.append(1)
            else:
                predict_label.append(0)
        score = EvaluationTool.cal_f1(np.array(predict_label), self.target_label)
        return score

    # Reproduction iteration
    def ga_generation(self, pop):
        # Selection phase
        pop_select = self.select(pop)
        # Crossover phase
        pop_cross = self.crossover(pop_select)
        # Mutation stage
        pop_mutation = self.mutation(pop_cross)
        return np.array(pop_mutation)

    # select
    # 20180807 only modify to here, next time from here
    def select(self, pop):
        fit_list, q_list = [], []  # Fitness list, cumulative probability list
        choose_index = set()  # Selected individuals
        fit_sum = sum(self.pop_f1_list)   # Sum of fitness
        p_list_array = np.divide(np.array(self.pop_f1_list), fit_sum)  # List of probability inherited to the next generation
        for i in range(len(p_list_array)):
            # Calculate the cumulative probability of each individual
            q = 0
            for j in range(i+1):
                q += p_list_array[j]
            q_list.append(q)
        # Generate a list of random numbers for selection
        rand_list = np.random.rand(pop.shape[0])
        for i in range(len(rand_list)):
            choose_index.add(self.get_index_from_list(rand_list[i], q_list))
        pop_new = [pop[i] for i in choose_index]
        return np.array(pop_new)

    # From the target list, return the position corresponding to the incoming parameter, the incoming parameter should be between the two values
    @staticmethod
    def get_index_from_list(num, target_list):
        for i in range(len(target_list)):
            if i == 0 and num <= target_list[0]:
                return 0
            else:
                if target_list[i-1] < num <= target_list[i]:
                    return i

    # cross
    def crossover(self, pop):
        son_list = []
        pair_num = int(pop.shape[0]/2)
        for i in range(pair_num):
            rand_num = random.random()
            if rand_num < self.crossover_rate:
                # Random selection of crossover locations
                rand_cross_index = random.randint(0, pop.shape[1]-1)  # ???Whether minus one
                # Cross and produce new offspring
                parent_a = pop.copy()[i*2, :]
                parent_b = pop.copy()[i*2+1, :]
                temp_parent_a = parent_a.copy()[rand_cross_index:]
                parent_a[rand_cross_index:] = parent_b[rand_cross_index:]  # New offspring a
                parent_b[rand_cross_index:] = temp_parent_a  # New offspring b
                son_list.append(parent_a)
                son_list.append(parent_b)
        if len(son_list) != 0:
            pop_new = np.append(pop, np.array(son_list), axis=0)
        else:
            pop_new = pop
        return pop_new

    # Mutations
    def mutation(self, pop):
        son_list = []
        for i in range(pop.shape[0]):
            rand_num = random.random()
            if rand_num < self.mutation_rate:
                # Randomly generate mutation positions
                rand_mutation_index = random.randint(0, pop.shape[1]-1)  # ???Whether minus one
                # Mutation produces new offspring
                parent = pop.copy()[i, :]
                if rand_mutation_index == pop.shape[1]-1:
                    # Last column mutation
                    r = random.random()*(self.genes_num-1)
                    parent[rand_mutation_index] = r
                else:
                    # Other column variants
                    r = random.random()
                    parent[rand_mutation_index] = r
                son_list.append(parent)
        if len(son_list) != 0:
            pop_new = np.append(pop, np.array(son_list), axis=0)
        else:
            pop_new = pop
        return pop_new
