from json import loads
from hatebase import HatebaseAPI

hatebase = HatebaseAPI({"key": '86ad75938e1e915f3598fde06df275b9'	})
filters = {'vocabulary': 'nigga', 'language': 'eng'}
output = "json"
query_type = "sightings"
response = hatebase.performRequest(filters, output, query_type)

# convert to Python object
response = loads(response)
