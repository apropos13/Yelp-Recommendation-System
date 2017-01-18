import json
import config_paths

import time
start_time = time.time()

business_path=config_paths.BUSINESS_PATH
tip_path= config_paths.TIP_PATH

def parse_user():

	user_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open('user.json') as j_file:
		user_data= json.load(j_file)

	print len(user_data)
	print 'userId= %s' %(user_data[0]['user_id'])
	print 'Name= %s' %(user_data[0]['name'])
	print 'Review Count= %s' %(user_data[0]['review_count'])
	print 'Avg Stars= %s' %(user_data[0]['average_stars'])
	print 'Votes = %s' %(user_data[0]['votes'])
	print 'Yelping Since = %s' %(user_data[0]['yelping_since'])
	print 'Compliments = %s' %(user_data[0]['compliments'])
	print 'Fans = %s' %(user_data[0]['fans'])
#print 'Friends= %s' %(user_data[0]['friends'])
#print elite?	

def parse_review():

	review_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open('review.json') as j_file:
		review_data= json.load(j_file)

	print len(review_data)
	#print 'Stars= %s' %(review_data[0]['stars'])
	#print 'text= %s' %(review_data[0]['text'])
	#print 'votes= %s' %(review_data[0]['votes'])
	print 'categs= %s' %(review_data[0]['categories'][0] )


def parse_checkIn():

	checkIn_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open('checkin.json') as j_file:
		checkIn_data= json.load(j_file)
		
	print len(checkIn_data)
	print 'Business id= %s' %(checkIn_data[0]['business_id'])
	print 'Info= %s' %(checkIn_data[0]['checkin_info'])

def parse_business():

	business_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open('business.json') as j_file:
		business_data= json.load(j_file)
		
	print len(business_data)
	print 'Business id= %s' %(business_data[0]['business_id'])
	print 'Info= %s' %(business_data[0]['name'])
	print 'City= %s' %(business_data[0]['city'])
	print 'State= %s' %(business_data[0]['state'])


if __name__=='__main__':
	#parse_user()
	parse_review()

	print("--- %s seconds ---" % (time.time() - start_time))

	'''
	my machine with this approach takes parse_user() down 
	from 52secs to 16secs. Still better approaches using json-stream.
	'''





