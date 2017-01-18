Restaurant Recommender System for Yelp


The original JSON files are not part of the Git repository because of their prohibiting size. In case of using a new dataset, the data needs to be parsed first. Otherwise, the recommender can be run straight away, using the pre-generated rating matrix from the provided Yelp JSON files.

1.) Parsing the data:
	(!) Skip to part 2.) to run the recommender with the relevant data already extracted and in a format friendly for the recommender module. Otherwise continue with the following steps.
	
	-> In the config_paths.py file, set the correct paths to the data to be parsed.
	-> Run crt_good_json.py to reformat the JSON files.
	-> [Requires the business.json file created in the previous step.] Run delete_from_bus.py to create a JSON file with only the restaurants filtered from the businesses.
	-> [Requires the review.json, user.json, restaurants.json files from the previous steps.] Run create_UserRating.py to produce the sparse matrix (ratingMatrix.npz) representing the user ratings, along with the files itemMapping.json and userMapping.json which contain the mapping from the original restaurant and user IDs, respectively, to the indices in the rating matrix.
	
2.) Running the recommender:
	(!) Requires the following data files to be present in the root folder: itemMapping.json, ratingMatrix.npz, restaurants.json, userMapping.json.
	
	-> In order to adjust the parameters for training, such as the number of latent factors for the SVD, the learning rates, the regularization coefficients, or the ratio of the training and the test data, change the respective values at the beginning of the Recommender class in recommender.py.
	-> Run main.py to run the program. It starts with the training of the model (using a training subset of the ratings in the rating matrix), and then the learned model is tested on the remaining data, corresponding to the test set. Informative statistics, including timestamps, are displayed at different stages.
		-> If it is desired to test one of the simpler prediction methods, uncomment the line calling the respective test() method in the testing section, while commenting out the other two test() calls. Subsequently comment out the line containing rec_system.train() in the training section, since the training phase is not utilized for them.
		-> Finally, a recommendation is performed for a specific user as an example, and the names of the restaurants recommended for this user, including the predicted ratings, are displayed.

The Report folder contains the Latex source files for the project report. The UnitTest folder contains the source code for a simple unit test. The produce_scatter.py file serves for plotting the acquired data for the report.
	