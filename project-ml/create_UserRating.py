import json
import config_paths
import numpy as np
from scipy.sparse import *

import time
start_time = time.time()

#checks if a given business is a restaurant 
def is_restaurant(a_dict, key):
	if key in a_dict:
		return True
	else:
		return False

def create_dict():
    reviewFile = []
    restFile = []
    user_ctr = 0
    item_ctr = 0
    all_users = {}
    all_items = {}
    user_indices = []
    item_indices = []
    ratings = []


	#load json
    with open('review.json') as j_file:
        reviewFile = json.load(j_file)

    with open('restaurants.json') as j_file:
        restFile = json.load(j_file)
    
    for review in reviewFile:
        b_id = review['business_id']

        #consider restaurants only
        if is_restaurant(restFile, b_id):
            u_id = review['user_id']
            num_stars = review['stars']

			#build the dict of users (so as to map their IDs to values 0,...)
            if u_id not in all_users:
                all_users[u_id] = user_ctr
                u_idx = user_ctr
                user_ctr += 1
            else:
                u_idx = all_users[u_id]
   
            #build the dict of items (so as to map their IDs to values 0,...)
            if b_id not in all_items:
                all_items[b_id] = item_ctr
                b_idx = item_ctr
                item_ctr += 1
            else:
            	b_idx = all_items[b_id]
            
            user_indices.append(u_idx)
            item_indices.append(b_idx)
            ratings.append(num_stars)
    
    print('[%.2f] Building the sparse matrix...' % (time.time() - start_time))

    #sort the ratings by users
    tuples = zip(ratings, user_indices, item_indices)
    ratings_sorted, user_indices_sorted, item_indices_sorted = zip(*sorted(tuples, key=lambda x: x[1]))

    #build a sparse matrix from the extracted ratings
    rating_matrix = coo_matrix((ratings_sorted, (user_indices_sorted, item_indices_sorted)), shape=(user_ctr, item_ctr), dtype=np.uint8)
    
    print('[%.2f] Saving the sparse matrix...' % (time.time() - start_time))

    np.savez('ratingMatrix', data = rating_matrix.data, row = rating_matrix.row, col = rating_matrix.col, shape = rating_matrix.shape)
    #np.savez('ratingMatrix', data = rating_matrix.data, indices = rating_matrix.indices, indptr = rating_matrix.indptr, shape = rating_matrix.shape)
    
    print('[%.2f] Saving the mappings...' % (time.time() - start_time))

    with open('userMapping.json', 'w') as userOut:
    	json.dump(all_users, userOut, sort_keys=True, indent=4, separators=(',', ': ') )

    with open('itemMapping.json', 'w') as itemOut:
    	json.dump(all_items, itemOut, sort_keys=True, indent=4, separators=(',', ': ') )


if __name__=='__main__':
    create_dict()
    
    #Just out of curiosity: Panos's machine takes 4min
    print("--- %.2f seconds ---" % (time.time() - start_time))

