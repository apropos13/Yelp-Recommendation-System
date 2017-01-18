import json
import config_paths

#simply searches for any restaurants in the array of categories
def exists_rest(arr):
	for element in arr:
		if (element=='Restaurants') or (element=='restaurants'):
			return True

	return False 

def delete():

	restaurant_dict={}

	#open json file
	with open('business.json') as j_file:
		businessFile= json.load(j_file)

	for business in businessFile:
		if exists_rest( business['categories']):
			id = business['business_id'] #take bussines id
			restaurant_dict[id]={} #delcare new dict
			restaurant_dict[id]['name']=business['name'] #insert value
			#in case we need to add extra values to the dict we would for example do:
			#restaurant_dict[id]['name']=business['attributes']


	with open('restaurants.json', 'w') as outfile:
		json.dump(restaurant_dict, outfile, sort_keys=True, indent=4, separators=(',', ': ') )



if __name__=='__main__':
	delete()

