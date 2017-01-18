from __future__ import division
import json
import numpy

class Recommender():
    
    def __init__(self, ):
        #constants
        self.num_user_factors = 2           #dimension of the latent factor vector
        self.num_item_factors = 2           #dimension of the latent factor vector
        self.default_factor_value = 0.00001 #initial value of the factors
        self.gamma2 = 0.001                 #learning rate (step)
        self.lambda7 = 0.015                #regularization constant
        self.step_size_factor = 0.9         #the rate of decreasing the step size in each iteration
        self.error_threshold = 10
        
        try:
            ratings_file = open('testUserRatings.json')
            #user_mapping_file = open('userMapping.json')
            #item_mapping_file = open('itemMapping.json')

            self.ratings = json.load(ratings_file)
            #self.users = json.load(user_mapping_file)
            #self.items = json.load(item_mapping_file)

            #hardcoded for this specific example
            self.num_users = 3
            self.num_items = 3

            #latent factor vectors (initialized with "guessed" values)
            self.user_factors = numpy.full((self.num_users, self.num_user_factors), self.default_factor_value)       #each row represents a P_u
            self.item_factors = numpy.full((self.num_items, self.num_item_factors), self.default_factor_value)       #each row represents a Q_i

            #bias vectors
            self.user_biases = numpy.zeros(self.num_users)
            self.item_biases = numpy.zeros(self.num_items)
            self.rating_avg = 0
        except IOError as ex:
            print('ERROR-----: ' + ex.strerror)
        else:
            ratings_file.close()
            #user_mapping_file.close()
            #item_mapping_file.close()


    def calculate_baseline_estimates(self):
        rating_ctr = 0
        item_ctrs = numpy.zeros(self.num_items)

        #iterate over all ratings to compute their average
        for user, item_list in self.ratings.iteritems():
            user_rating_sum = 0
            for item, rating in item_list.iteritems():
                user_rating_sum += rating       #note that users are handled in a different way than items due to the structure of the "rating matrix"

                self.item_biases[int(item)] += rating
                item_ctrs[int(item)] += 1

                self.rating_avg += rating
                rating_ctr += 1
            
            #calculate averages which will be turned into biases once the overall rating average is known
            if len(item_list) > 0:
                self.user_biases[int(user)] = user_rating_sum / len(item_list)
            else:
                self.user_biases[int(user)] = 0.0

        self.rating_avg /= rating_ctr

        #calculate the user biases
        self.user_biases -= self.rating_avg     #vector operation

        #calculate the item biases
        with numpy.errstate(divide='ignore', invalid='ignore'):
            self.item_biases = self.item_biases / item_ctrs - self.rating_avg     #vector operation
            self.item_biases[~numpy.isfinite(self.item_biases)] = 0         #convert inf and NaN (as a consequence of the division operation) values to 0

    def predict_rating(self, user, item):
        baseline_estimate = self.rating_avg + self.user_biases[user] + self.item_biases[item]
        factor_product = numpy.dot(self.user_factors[user, :], self.item_factors[item, :])

        #DEBUG PRINT
        #print(baseline_estimate)

        return baseline_estimate + factor_product

    def train(self):
        self.calculate_baseline_estimates()

        iter_ctr = 0
        while True:
            #reset the cumulative error
            error_sum = 0
            #loop over all the known ratings
            for user, item_list in self.ratings.iteritems():
                for item, rating in item_list.iteritems():
                    #loop over all dimensions of the factor vectors
                    for dim in range(self.num_user_factors):
                        #calculate the current error of the prediction
                        prediction = self.predict_rating(int(user), int(item))
                        error_user_item = rating - prediction
                        error_sum += error_user_item

                        #update the factors
                        self.user_factors[int(user), dim] += self.gamma2 * (error_user_item * self.item_factors[int(item), dim] - self.lambda7 * self.user_factors[int(user), dim])
                        self.item_factors[int(item), dim] += self.gamma2 * (error_user_item * self.user_factors[int(user), dim] - self.lambda7 * self.item_factors[int(item), dim])

            #if error_sum < self.error_threshold:
                #break

            #terminate after a fixed number of iterations
            if iter_ctr > 30:
                break
            else:
                iter_ctr += 1

            self.gamma2 *= self.step_size_factor

        #DEBUG PRINT
        print('user factors:\n' + str(self.user_factors))
        print('item factors:\n' + str(self.item_factors))
        print('')

    def test(self):
        print('r(1, 2) = ' + str(self.predict_rating(1, 2)) + '  [real rating: 2]')
        print('r(2, 1) = ' + str(self.predict_rating(2, 1)) + '  [real rating: 4]')