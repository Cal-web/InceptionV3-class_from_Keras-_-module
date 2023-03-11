from flask import Flask, render_template, request, jsonify,url_for, flash

app = Flask(__name__)

@app.route('/')
def upload_form():
     return render_template('cal.html')


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import urllib.request
from werkzeug.utils import secure_filename


model = tf.keras.applications.InceptionV3(weights='imagenet')
model.save('inception_v3.h5')

import cv2
import numpy as np

@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		return render_template('cal.html', filename=filename)



@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


# Define a function to extract frames from a video file
def extract_frames_from_video(video_file):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file)


    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file)

    # Loop through the video frames and append each one to the frames list
    while True:
        # Read a frame from the video file
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the frames list
        frames.append(frame)

    # Release the video file and return the frames list
    cap.release()
    return frames


# Define a function to detect objects in a list of frames
def detect_objects_in_video(frames):
    # Initialize an empty list to store the object detections
    object_detections = []

    # Loop through the frames and detect objects in each one
    for frame in frames:
        # Resize the frame to 299x299 pixels (the input size of the Inception V3 model)
        frame = cv2.resize(frame, (299, 299))

        # Preprocess the frame by subtracting the mean RGB values of the ImageNet dataset
        frame = preprocess_input(frame)

        # Pass the frame through the Inception V3 model to get the predicted probabilities for each object class
        predictions = model.predict(np.array([frame]))

        # Get the top 3 most likely object classes and their corresponding probabilities
        top_classes = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=3)[0]

        # Append the top object classes and their probabilities to the object detections list
        object_detections.append(top_classes)

    # Return the object detections list
    return object_detections


# Define a route to handle video uploads and object detection
@app.route('/detect_objects_in_video', methods=['POST'])
def detect_objects():
    # Check if a video file was uploaded
    if 'video' not in request.files:
        return 'No video file uploaded'

    # Load the video file from the request
    video_file = request.files['video']

    # Extract frames from the video file
    frames = extract_frames_from_video(video_file)

    # Detect objects in the video frames
    object_detections = detect_objects_in_video(frames)

    # Return the object detections as JSON
    return jsonify(object_detections)


# # Set the maximum allowed file size to 5 megabytes
# app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


# Route to handle file upload
@app.route('/video_upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    # Process the uploaded file here
    return 'File uploaded successfully'

if __name__ == "__main__":
    app.run(port = 5001)

