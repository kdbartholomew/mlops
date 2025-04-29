import requests

comment = {'reddit_comment':'Testing a comment.'}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=comment)
print(response.json())