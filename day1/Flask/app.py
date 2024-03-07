from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return "简易部署系统"


@app.route('/tt_params', methods=['GET', 'POST'])
@app.route('/tt_params/<string:name>/<int:age>', methods=['GET', 'POST'])
def tt_params(name="小明", age=20):
    if request.method == 'GET':
        _args = request.args
    elif request.method == 'POST':
        _args = request.form
    else:
        raise ValueError("仅支持GET和POST方法，当前异常实际上不会被触发")
    print(f'参数类型：{_args} -- {type(_args)}')
    address = _args.get('address', '默认地址为上海')
    return jsonify({
        'code': 200,
        'msg': '成功',
        'name': name,
        'age': age,
        'address': address
    })


