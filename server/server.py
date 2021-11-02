from flask import Flask

app = Flask(__name__)

@app.route("/")
def check_leaf():
	return "Analizing ..."

if (__name__ == "__main__"):
	app.run()
