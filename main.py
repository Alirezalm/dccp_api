import json

from dotenv import load_dotenv
from flask import Flask, jsonify, request
import os, sys

app = Flask(__name__)
PATH = os.path.join(os.path.dirname(__file__))
ENV_PATH = os.path.join(PATH, '.env')

load_dotenv(ENV_PATH)


@app.route('/', methods = ['POST', 'GET'])
def main_page():
    mpi_run = None
    if request.method == 'POST':
        problem_data = json.loads(request.get_data())
        with open('config.json', 'w') as jsonfile:
            json.dump(problem_data, jsonfile)

        # mpi_run = os.system(f"mpiexec -n {problem_data['nNodes']} {sys.executable} {PATH}/run.py")
        import run
        run.run(problem_data)
        if mpi_run == 0:
            with open('solution.json') as jsonfile:
                my_data = json.load(jsonfile)
            return jsonify(my_data)
        else:
            return jsonify({'status': 'failed'})
    else:
        return jsonify({'status': 'GET REQUEST NOT ALLOWED'})


if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0', port = '5000')
