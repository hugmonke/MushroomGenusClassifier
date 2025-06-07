import flask
import fastai.vision.all as fa
import os

app = flask.Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():

    if flask.request.method == 'POST':
        
        if 'mushroom' not in flask.request.files:
            return flask.redirect(flask.request.url)
        
        file = flask.request.files['mushroom']
        if file.filename == '':
            return flask.redirect(flask.request.url)

        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            flask.flash('Invalid file type. Accepted files are: .PNG, .JPG, .GIF, .WEBP', 'error')
            return flask.redirect(flask.request.url)
        
        if file:
            imagepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(imagepath)

            predicted_name, prediction_idx, probabilities = model.predict(imagepath)
            confidence = 100*float(probabilities[prediction_idx])
            
            return flask.render_template(
                                    'index.html', 
                                    prediction=predicted_name, 
                                    confidence=confidence,
                                    image_path=file.filename
                                    )
    
    return flask.render_template('index.html')

if __name__ == '__main__':
    model = fa.load_learner('model_1.pkl')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)