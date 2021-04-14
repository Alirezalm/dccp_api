import json

from dotenv import load_dotenv
from flask import Flask, jsonify, request
import os, sys

app = Flask(__name__)

PATH = os.path.join(os.path.dirname(__file__), '.env')

load_dotenv(PATH)


@app.route('/', methods = ['POST'])
def main_page():
    if request.method == 'POST':
        problem_data = json.loads(request.get_data())
        with open('config.json', 'w') as jsonfile:
            json.dump(problem_data, jsonfile)

        mpi_run = os.system(f"mpiexec -n {problem_data['nNodes']} {sys.executable} {PATH}/run.py")
        print(sys.path)
        with open('solution.json') as jsonfile:
            my_data = json.load(jsonfile)
    return jsonify(my_data)


if __name__ == '__main__':
    app.run(debug = True, host = '127.0.0.1', port = '5000')
