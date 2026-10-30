from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI(title="iris predict")
# 这里换成你train打印出来的run_id
RUN_ID = "2cfb4fc1a10a41e5a9570eb623b649a6"
# 容器内专用地址
mlflow.set_tracking_uri("http://host.docker.internal:5000")
MODEL_URI = f"runs:/{RUN_ID}/model"

model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/predict")
def predict(sepal_length:float, sepal_width:float, petal_length:float, petal_width:float):
    pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return {"prediction": int(pred[0])}

