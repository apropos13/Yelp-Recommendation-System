import numpy as np
import time
import recommender

if __name__=='__main__':
    start_time = time.time()

    # initalizing
    print('[%.2fs] Initializing...' % (time.time() - start_time))

    rec_system = recommender.Recommender()
    
    
    # training
    print('\nTraining data:')
    print('-> Number of ratings:  %s' % len(rec_system.ratings_train.data))
    print('-> Number of distinct users:  %s' % len(np.unique(rec_system.ratings_train.row)))
    print('-> Number of distinct items:  %s' % len(np.unique(rec_system.ratings_train.col)))
    print('-> Number of latent factors:  %d') %(rec_system.num_user_factors)

    print('\n[%.2fs] Training...' % (time.time() - start_time))

    rec_system.train()

    print('\nLearned values:')
    print('\n-> User factors:')
    print(rec_system.user_factors)
    print('\n-> Item factors:')
    print(rec_system.item_factors)
    
    
    # testing
    print('\nTest data:')
    print('-> Number of ratings:  %s' % len(rec_system.ratings_test.data))
    print('-> Number of distinct users:  %s' % len(np.unique(rec_system.ratings_test.row)))
    print('-> Number of distinct items:  %s' % len(np.unique(rec_system.ratings_test.col)))
    print('\n[%.2fs] Testing...' % (time.time() - start_time))

    test_rmse = rec_system.test()
    #test_rmse = rec_system.test_baseline()
    #test_rmse = rec_system.test_naive()

    print('\nRMSE: %s\n' % test_rmse)

    #rec_system.test_example()

    print('[%.2fs] Model successfully created. Ready to recommend some goodies!' % (time.time() - start_time))

    
    # recommendation of top items for a specific user
    user_id = 'PUFPaY9KxDAcGqfsorJp3Q'
    num_items = 5
    recommendations = rec_system.recommend(user_id, num_items)

    if len(recommendations) > 0:
        print('\n----------------------------------------------------------')
        print('Recommended restaurants for user "%s":' % user_id)
        print('----------------------------------------------------------\n')

        for rest_name, pred_rating in recommendations:
            print('-> %s  (%.2f)' % (rest_name, pred_rating))
        print('')
