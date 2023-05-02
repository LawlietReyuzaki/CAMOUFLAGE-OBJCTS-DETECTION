import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        return redirect(request.url)

    # Check if the file is a valid image format
    if not allowed_file(file.filename):
        return redirect(request.url)

    # Save the file to the uploads folder
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Load the original image and the mask
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    mask = cv2.imread('mask.png', 0)

    # Apply morphological dilation to the mask to make the highlighted area thicker
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    # Use the mask to create a binary mask
    ret, binary_mask = cv2.threshold(dilation, 1, 255, cv2.THRESH_BINARY)

    # Convert the binary mask to color
    color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Blend the original image and the color mask together
    highlighted_img = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    # Save the mask and highlighted image to the static folder
    cv2.imwrite(os.path.join('static', 'mask.jpg'), binary_mask)
    cv2.imwrite(os.path.join('static', 'highlighted.jpg'), highlighted_img)

    # Display the results
    return render_template('result.html', filename=filename)

# Utility function to check if a file is a valid image format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == '__main__':
    app.run(debug=True)
