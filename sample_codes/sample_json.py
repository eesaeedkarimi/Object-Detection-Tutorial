import json
with open('../sample_files/sample_json.json', 'r') as f:
    data = json.load(f)
print(data['attribute1'])
print(data['attribute2'])
print(data['attribute3'])
print(data['attribute4'])
print(data['attribute4'][0]['attribute4_1_1'])
