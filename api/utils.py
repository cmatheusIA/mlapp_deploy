import pickle
import pandas as pd
import json


def load_pickle(path):
    with open(path, 'rb') as arq:
        model_pkl = pickle.load(arq)
    return model_pkl


def bytes_to_json(data):
    converted = data.decode('utf8').replace("'", '"')
    bd = json.loads(converted)
    return bd


def create_data(data):
    if data:
        data = bytes_to_json(data)

        if isinstance(data, dict):
            df = pd.DataFrame(data, index=[0])
        else:
            df = pd.DataFrame(data, columns=data[0].keys)
    return df


def preprocess(df):
    le = load_pickle('./models/encoder.pkl')
    for col in df.columns:
        if df[col].dtypes == object:
            df[col] = le.transform(df[col])
    return df


def predict_pipe(data):
    df = create_data(data)
    data_final = create_data(data)

    data_pre = preprocess(df)
    model = load_pickle('./models/model_randFlorest.pkl')

    pred = model.predict(data_pre)
    proba = model.predict_proba(data_pre)
    data_final['prediction'] = pred
    data_final['probabilidade_fraude'] = round(proba[0][1] * 100, 2)
    data_final['probabilidade_nao_fraude'] = round(proba[0][0] * 100, 2)

    return data_final.to_json(orient='records')
