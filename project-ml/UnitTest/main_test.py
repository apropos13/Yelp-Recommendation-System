import recommender_test
#import rmse_test

if __name__=='__main__':
    #initalizing
    recSystem = recommender_test.Recommender()
    
    #rmseCalculator = rmse_test.RmseCalculator()
    
    #training
    print('Training...')
    recSystem.train()

    #testing
    print('Testing...')
    recSystem.test()

    #rmse
    #print recSystem.predict_rating(200, 12)
    #print('rmse: ')
   # print rmseCalculator.calculateRMSE(recSystem)
