import requests

url = 'http://121.40.96.93:9999/predict?features=4,5,6,8'
# url = 'http://127.0.0.1:9999/predict?features=1,2,3,4;0.1,0.2,0.5,0.3;2.1,3.5,1.2,1.3'
response = requests.get(url)
if response.status_code == 200:
    result = response.json()
    if result['code'] != 0:
        print("调用服务异常!")
    print(result)
    print(type(result))
print(response)