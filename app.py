from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline


application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Company=request.form.get('company'),
            TypeName=request.form.get('typename'),
            
            Inches= float(request.form.get("inches")),
            Cpu= request.form.get("cpu"),
            Memory=request.form.get("memory"),
            Screen_height=float(request.form.get("screen_height")),
            Screen_width=float(request.form.get("screen_width")),
            OpSys=request.form.get("opsys"),
            Ram=float(request.form.get("ram")),
            Weight=float(request.form.get("weight")),
        )
        pred_df = data.convert_to_dataFrame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)