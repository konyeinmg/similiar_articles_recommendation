import json
import numpy as np 

def readData(url):
    f = open(url)
    data = json.load(f)
   
    contents = []
    #only get contents
    for item in data:
        contents.append(item.get('content'))

    return contents