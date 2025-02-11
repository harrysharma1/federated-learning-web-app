import base64
from io import BytesIO
from flask import Flask, render_template
from torchvision import datasets

app = Flask(__name__)
dst = datasets.CIFAR100("~/.torch", download=True)

@app.route("/")
def index():
    images = []
    for idx in range(25):
        img = dst[idx][0]
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        images.append(img_str)
        
    return render_template("index.html", images = images)
 
if __name__ == '__main__':
	app.run(debug=True)