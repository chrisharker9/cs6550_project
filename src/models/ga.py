import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GA(object):
    def __init__(self, 
                 mutation_rate=0.1,
                 cross_over=False,
                 pool_size=2,
                 w=None,
                 n_elites=0,
                 max_iter=10,
                 log_file_name='fitness_log.csv',
                 population_file_name='population.csv',
                 verbose=False):
        self.mutation_rate = mutation_rate
        self.cross_over = cross_over
        self.pool_size = pool_size
        self.w = w
        self.n_elites = n_elites
        self.max_iter = max_iter
        self.log_file_name = log_file_name
        self.population_file_name = population_file_name
        self.verbose = verbose
        
        self.__scaler1 = MinMaxScaler()
        self.__scaler2 = MinMaxScaler()
        
    def fit(self, X=None, P=None, targeted_items=None):
        
        if self.pool_size % 2 != 0:
            raise ValueError('Parameter pool_size must be an even number.')
        
        n_users, n_items = X.shape
        #print('n_users: ', n_users, 'n_items: ', n_items)
        
        # Initialize Random Population
        if P is None:
            #P = np.random.choice(np.array([0., 1., 2., 3., 4., 5.]),
            #                     p=np.array([0.93695331, 0.00385215, 0.00716841, 0.01711402, 0.02154558, 0.01336653]),
            #                     size=(n_users, n_items))
            P = np.zeros((n_users, n_items))
            
            means = np.zeros((n_items, 6))
            for j in range(n_items):
                cnts = np.unique(X[:, j], return_counts=True)
                means[j, cnts[0].astype(int)] = cnts[1]
            means = (means/n_users).T
            
            for j in range(n_items):
                P[:, j] = np.random.choice(np.array([0., 1., 2., 3., 4., 5.]),
                                       p=means[:, j],
                                       size=n_users)
        
        pF = self.__evaluate_fitness(X, P, targeted_items)
        #F, F1, F2 = self.__evaluate_fitness(X, P, targeted_items)
        #for a,b in zip(F1,F2): print(a,b)
        n_iter = 0
        while n_iter < self.max_iter:
            # Select Parents, proportional to F
            parents_idx = np.random.choice(n_users, size=self.pool_size, replace=False)
            pool = P[parents_idx, :].copy()

            # cross-over
            if self.cross_over:
                for i in range(0, len(parents_idx)-1, 2):
                    #print(i, parents_idx[i], parents_idx[i+1])
                    parent1 = X[parents_idx[i], :]
                    parent2 = X[parents_idx[i+1], :]

                    child1 = pool[i]
                    child2 = pool[i+1]

                    # choose random index to crossover at
                    ridx = np.random.randint(low=1, high=n_items+1)
                    tmp = child2[:ridx].copy()
                    child2[:ridx], child1[:ridx] = child1[:ridx], tmp

            # mutate children. 
            for c in range(len(pool)):
                mask = np.random.binomial(n=1, p=self.mutation_rate, size=n_items)
                for j in range(n_items):
                    if mask[j] == 1:
                        pool[c, j] = np.random.choice(np.array([0., 1., 2., 3., 4., 5.]),
                                                 p=means[:, j],
                                                 size=1)

            # evaluate function of children
            cF = self.__evaluate_fitness(X, pool, targeted_items)

            # replacement
            # for each child, choose one individual from the population
            # randomly. Compare fitness functions. If child is more fit,
            # replace individual in population with child (unless random
            # individual is in top x individuals, the draw again)
            for c in range(len(pool)):
                # pick random number
                ridx = np.random.randint(low=0, high=n_users)
                #print(cF[c], pF[ridx], cF[c][0] > pF[ridx][0], cF[c][1] > pF[ridx][1] )
                #print((cF[c] >= pF[ridx]).all(), (cF[c] > pF[ridx]).any())
                if (cF[c] >= pF[ridx]).all() & (cF[c] > pF[ridx]).any():
                    pF[ridx] = cF[c]
                    P[ridx] = pool[c]

            with open(self.log_file_name, 'a+') as f:
                f.write(f"{n_iter}, {(-1*pF[:, 0]).min()}, {(-1*pF[:, 0].max())}, {(-1*pF[:, 0]).mean()}, {pF[:, 1].min()}, {pF[:, 1].max()}, {pF[:, 1].mean()}\n")
            
            #print(F"iter: {n_iter}, F1 Min: {(-1*pF[:, 0]).min()}, F1 Mean: {(-1*pF[:, 0]).mean()}, F2 Max: {pF[:, 1].max()}, F2 Mean: {pF[:, 1].mean()}")
            n_iter += 1
            if self.verbose:
                if n_iter % 10000 == 0:
                    print(f"n_iter: {n_iter}")
        np.savetxt(self.population_file_name, P, delimiter=",")  
            
            
        return self
    
    def __evaluate_fitness(self, X=None, P=None, targeted_items=None):
        """
        Params
        ------
        X : ndarray[n_users, n_items]
            training set
        P : ndarray[n_users, n_items]
            Population
        """
        # F1 is some fitness that reflects its similarity to other
        # users in the initial population
        mask = np.ones(X.shape[1], dtype=bool)
        mask[targeted_items] = False
        distances = pairwise_distances(X[:, mask], P[:, mask], 'l1') # X by P
        F1 = distances.min(axis=0) # take min distance (max similarity)
        
        # F2 is some fitness that reflects how much it affects 
        # the targeted ratings. 
        mean_ratings = X[:, targeted_items].mean(axis=0) # Mean rating for each targeted item
        ratings_P = P[:, targeted_items] 
        
        F2 = (ratings_P - mean_ratings).mean(axis=1) # The the Fake users rating higher or lower than the items mean rating
        
        F = np.stack([-1*F1, F2], axis=1)
        
        return F
            

        
        
        