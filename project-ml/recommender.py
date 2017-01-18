from __future__ import division
import json
import random
import math
import numpy as np
import scipy.sparse as sps

class Recommender():
    
    def __init__(self):
        # parameters
        self.num_user_factors = 4           # dimension of the latent factor vector
        self.num_item_factors = 4           # dimension of the latent factor vector
        self.default_factor_value = 0.1     # initial value of the factors
        self.use_noise = False              # indicator whether noise should be added to the initial factor values
        self.noise_variance = 0.01          # variance of the noise

        self.gamma1 = 0.04                  # learning rate for biases (step) [paper: 0.007]
        self.gamma2 = 0.04                  # learning rate for factors (step) [paper: 0.007]
        self.lambda6 = 0.8                  # regularization constant for biases [paper: 0.005]
        self.lambda7 = 0.8                  # regularization constant for factors [paper: 0.015]
        self.step_size_factor = 0.9         # the rate of decreasing the step size in each iteration [paper: 0.9]
        
        self.training_proportion = 0.8      # proportion of data to be used for training
        self.num_ratings_cap = 1e5          # a cap for the data to be used
                                                # note #1: delete ratingMatrixTrain.npz and ratingMatrixTest.npz to generate them anew
                                                # note #2: not used if the first condition in the split_dataset() method is commented out

        # fix the seed to work with the same "random" numbers on every run
        random.seed(6)


        with open('userMapping.json') as user_mapping_file, open('itemMapping.json') as item_mapping_file, np.load('ratingMatrix.npz') as sparse_matrix:
            self.users = json.load(user_mapping_file)
            self.items = json.load(item_mapping_file)
            self.ratings = sps.coo_matrix((sparse_matrix['data'], (sparse_matrix['row'], sparse_matrix['col'])), shape = sparse_matrix['shape'])

        self.num_users = len(self.users)
        self.num_items = len(self.items)

        # latent factor vectors (initialized with "guessed" values)
        self.user_factors = np.full((self.num_users, self.num_user_factors), self.default_factor_value)             # each row represents a P_u
        self.item_factors = np.full((self.num_items, self.num_item_factors), self.default_factor_value)             # each row represents a Q_i
        
        # add Gaussian noise to the factors
        if self.use_noise:
            mean = np.zeros(self.num_user_factors)
            covariance = np.diag(self.noise_variance * np.ones(self.num_user_factors))
            self.user_factors += self.default_factor_value * np.random.multivariate_normal(mean, covariance, self.num_users)
            self.item_factors += self.default_factor_value * np.random.multivariate_normal(mean, covariance, self.num_items)
        
        # bias vectors
        self.user_biases = []
        self.item_biases = []
        self.rating_avg = 0

        # if the training and the test dataset have not been generated, generate them now
        try:
            with np.load('ratingMatrixTrain.npz') as training_matrix, np.load('ratingMatrixTest.npz') as test_matrix:
                self.ratings_train = sps.coo_matrix((training_matrix['data'], (training_matrix['row'], training_matrix['col'])), shape = training_matrix['shape'])
                self.ratings_test = sps.coo_matrix((test_matrix['data'], (test_matrix['row'], test_matrix['col'])), shape = test_matrix['shape'])
        except:
            self.split_dataset(self.training_proportion, self.num_ratings_cap)

    
    def split_dataset(self, training_proportion, num_ratings_cap):
        rating_ctr = 0
        
        data_train, row_train, col_train = ([], [], [])
        data_test, row_test, col_test = ([], [], [])

        for rating, row, col in zip(self.ratings.data, self.ratings.row, self.ratings.col):
            # use a cap to limit the sizes of the datasets
            #if rating_ctr >= num_ratings_cap:
                #break

            # divide the datapoints into a training set and a test set (keeping a certain ratio)
            if random.uniform(0, 1) < training_proportion:
                data_train.append(rating)
                row_train.append(row)
                col_train.append(col)
            else:
                data_test.append(rating)
                row_test.append(row)
                col_test.append(col)
                
            rating_ctr += 1
        
        # build the rating matrices for the training and the test dataset
        self.ratings_train = sps.coo_matrix((data_train, (row_train, col_train)), dtype=np.int8)
        self.ratings_test = sps.coo_matrix((data_test, (row_test, col_test)), dtype=np.int8)
        
        np.savez('ratingMatrixTrain', data = self.ratings_train.data, row = self.ratings_train.row, col = self.ratings_train.col, shape = self.ratings_train.shape)
        np.savez('ratingMatrixTest', data = self.ratings_test.data, row = self.ratings_test.row, col = self.ratings_test.col, shape = self.ratings_test.shape)

    def calculate_baseline_estimates(self):
        # extract the non-zero elements from the rating matrix
        rows, cols, vals = sps.find(self.ratings_train)

        # compute the overall average rating
        self.rating_avg = vals.mean()

        # compute the average ratings for each user
        per_user_counts = np.bincount(rows)
        per_user_sums = np.bincount(rows, weights=vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_user_averages = per_user_sums / per_user_counts
        self.user_biases = per_user_averages - self.rating_avg
        self.user_biases[~np.isfinite(self.user_biases)] = 0.0
        # fill the rest of the vector with zeros
        self.user_biases = np.append(self.user_biases, np.zeros(self.num_users - len(self.user_biases)))
        
        # compute the average ratings for each item
        per_item_counts = np.bincount(cols)
        per_item_sums = np.bincount(cols, weights=vals)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_item_averages = per_item_sums / per_item_counts
        self.item_biases = per_item_averages - self.rating_avg
        self.item_biases[~np.isfinite(self.item_biases)] = 0.0
        # fill the rest of the vector with zeros
        self.item_biases = np.append(self.item_biases, np.zeros(self.num_items - len(self.item_biases)))

    def predict_rating(self, user, item):
        baseline_estimate = self.rating_avg + self.user_biases[user] + self.item_biases[item]
        factor_product = np.dot(self.user_factors[user, :], self.item_factors[item, :])

        # DEBUG PRINT
        #print(baseline_estimate)

        return baseline_estimate + factor_product

    def train(self):
        max_num_iters = 30
        iter_ctr = 0

        self.calculate_baseline_estimates()

        while True:
            # loop over all the known ratings
            for rating, user, item in zip(self.ratings_train.data, self.ratings_train.row, self.ratings_train.col):
                # loop over all dimensions of the factor vectors
                for dim in range(self.num_user_factors):
                #for dim in range(0, random.randint(1, self.num_user_factors)):      # stochastic
                    #calculate the current error of the prediction
                    prediction = self.predict_rating(user, item)
                    error_user_item = rating - prediction

                    # update the factors
                    self.user_factors[user, dim] += self.gamma2 * (error_user_item * self.item_factors[item, dim] - self.lambda7 * self.user_factors[user, dim])
                    self.item_factors[item, dim] += self.gamma2 * (error_user_item * self.user_factors[user, dim] - self.lambda7 * self.item_factors[item, dim])

                prediction = self.predict_rating(user, item)
                error_user_item = rating - prediction

                # update the biases
                self.user_biases[user] += self.gamma1 * (error_user_item - self.lambda6 * self.user_biases[user])
                self.item_biases[item] += self.gamma1 * (error_user_item - self.lambda6 * self.item_biases[item])

            # terminate after a fixed number of iterations
            iter_ctr += 1
            if iter_ctr > max_num_iters:
                break
            
            # decay the learning rate
            #self.gamma1 *= self.step_size_factor
            #self.gamma2 *= self.step_size_factor


    def test(self):
        rmse = 0.0

        # calculate the root-mean-square error on the test dataset
        for rating, user, item in zip(self.ratings_test.data, self.ratings_test.row, self.ratings_test.col):
            prediction = self.predict_rating(user, item)

            # apply an upper/lower cap to the predictions
            if prediction > 5.0:
                prediction = 5.0
            elif prediction < 1.0:
                prediction = 1.0

            rmse += (rating - prediction)**2

        rmse /= len(self.ratings_test.data)
        rmse = math.sqrt(rmse)

        return rmse

    def test_baseline(self):
        rmse = 0.0

        self.calculate_baseline_estimates()

        # calculate the root-mean-square error on the test dataset using the average rating as prediction
        for rating, user, item in zip(self.ratings_test.data, self.ratings_test.row, self.ratings_test.col):
            prediction = self.rating_avg + self.user_biases[user] + self.item_biases[item]

            # apply an upper/lower cap to the predictions
            if prediction > 5.0:
                prediction = 5.0
            elif prediction < 1.0:
                prediction = 1.0

            rmse += (rating - prediction)**2

        rmse /= len(self.ratings_test.data)
        rmse = math.sqrt(rmse)

        return rmse

    def test_naive(self):
        rmse = 0.0

        self.calculate_baseline_estimates()
        prediction = self.rating_avg

        # calculate the root-mean-square error on the test dataset using the baseline estimates as prediction
        for rating, user, item in zip(self.ratings_test.data, self.ratings_test.row, self.ratings_test.col):
            rmse += (rating - prediction)**2

        rmse /= len(self.ratings_test.data)
        rmse = math.sqrt(rmse)

        return rmse

    def test_example(self):
        print('r(200, 12) = ' + str(self.predict_rating(200, 12)) + '  [real rating: 1]')
        print('r(655, 13281) = ' + str(self.predict_rating(655, 13281)) + '  [real rating: 2]')
        print('r(806, 26152) = ' + str(self.predict_rating(806, 26152)) + '  [real rating: 3]')
        print('r(14, 19222) = ' + str(self.predict_rating(14, 19222)) + '  [real rating: 4]')
        print('r(365, 16511) = ' + str(self.predict_rating(365, 16511)) + '  [real rating: 5]')

    
    def recommend(self, user_id, num_items):
        # extract the user's ID
        try:
            user = self.users[user_id]
        except KeyError:
            print('\nError: Unknown User ID. Please, provide a valid ID for the recommendation to work.\n')
            return ()

        user_rated_items = self.ratings.tocsr().getrow(user).indices

        # predict ratings of all items for the given user
        baseline_estimates = self.rating_avg + self.user_biases[user] + self.item_biases
        factor_products = np.dot(self.user_factors[user, :], self.item_factors.transpose())
        predictions = baseline_estimates + factor_products

        # sort the items by their predicted rating, and ignore the items already rated by the user
        prediction_tuples = zip(sorted(self.items.values()), predictions)
        predictions_unrated = [(i, p) for (i, p) in prediction_tuples if i not in user_rated_items]
        predictions_sorted = sorted(predictions_unrated, key=lambda x: x[1], reverse=True)
        
        # filter the top picks only and apply an upper cap on the predicted ratings
        top_items, top_predictions = zip(*predictions_sorted)
        top_items = top_items[0:num_items]
        top_predictions = (np.fmin(top_predictions, 5.0))[0:num_items]

        with open('restaurants.json') as restaurants_file:
            restaurant_data = json.load(restaurants_file)

        # extract the names of the recommended restaurants
        top_rest_names = []
        for item in top_items:
            for item_id, item_idx in self.items.iteritems():
                if item_idx == item:
                    top_rest_names.append(restaurant_data[item_id]['name'])
                    break

        if len(top_rest_names) < num_items:
            print('\nError: Unknown Restaurant ID.\n')
            return ()
        
        # return the top num_items recommendations
        return zip(top_rest_names, top_predictions)
