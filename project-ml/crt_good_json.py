import json
import config_paths
import time


start_time = time.time()

tip_path= config_paths.TIP_PATH

def create_user(user_path=config_paths.USERS_PATH):

	user_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open(user_path) as f:
		for line in f:
			user_data.append(json.loads(line))

	with open('user.json', 'w') as outfile:
		json.dump(user_data, outfile)
	

def create_review(review_path=config_paths.REVIEW_PATH):

	review_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open(review_path) as f:
		for line in f:
			review_data.append(json.loads(line))

	with open ('review.json','w') as outfile:
		json.dump(review_data, outfile, sort_keys=True, indent=4, separators=(',', ': ') )


def create_checkIn(checkIn_path=config_paths.CHECKIN_PATH):

	checkIn_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open(checkIn_path) as f:
		for line in f:
			checkIn_data.append(json.loads(line))

	with open('checkin.json', 'w') as outfile:
		json.dump(checkIn_data, outfile, sort_keys=True, indent=4, separators=(',', ': ') )


def create_business(business_path=config_paths.BUSINESS_PATH):

	business_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open(business_path) as f:
		for line in f:
			business_data.append(json.loads(line))

	with open('business.json', 'w') as outfile:
		json.dump(business_data, outfile, sort_keys=True, indent=4, separators=(',', ': ') )


def create_tip(tip_path= config_paths.TIP_PATH):

	tip_data=[]

	#need to parse file line by line and use json.load to parse individual strings
	with open(tip_path) as f:
		for line in f:
			tip_data.append(json.loads(line))

	with open('tip.json', 'w') as outfile:
		json.dump(tip_data, outfile, sort_keys=True, indent=4, separators=(',', ': ') )


if __name__=='__main__':
	create_user()
	create_review()
	#create_checkIn()
	create_business()
	#create_tip()

	#my machine --- 408 seconds apprx 7 min
	print("--- %s seconds ---" % (time.time() - start_time))
