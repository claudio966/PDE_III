from flask import Flask, request, abort, redirect, url_for

app = Flask(__name__)

@app.route("/leaf", methods=['post'])
def upload_file():
	if request.method == 'POST':
		f = request.files['the_file']
    	f.save('/var/www/Uploads/uploaded_file.txt')

if (__name__ == "__main__"):
	app.run()
