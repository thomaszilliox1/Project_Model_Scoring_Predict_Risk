from pydantic.dataclasses import dataclass
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import FastAPI, Path, HTTPException, Body
from sklearn.metrics import confusion_matrix
import mlflow
import uvicorn

ZIP_TEST_DATA_FILENAME = "test_data_2.zip"
MLFLOW_MODEL_FOLDER = "mlflow_model"
BEST_THRESHOLD = 0.27

@dataclass
class Client_credit:
    SK_ID_CURR: int
    FLAG_OWN_REALTY: int
    FLAG_OWN_CAR: int
    OWN_CAR_AGE: float
    NAME_INCOME_TYPE_Working: bool
    DAYS_EMPLOYED: float
    AMT_GOODS_PRICE: float
    AMT_CREDIT_SUM_mean: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    PRED_PROBA: float = 0
    PRED_TARGET: int = 0
    TARGET: bool = 0

    def to_new_data(self):
        new_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == self.SK_ID_CURR].copy()
        # Mettre à jour les colonnes avec les types natifs
        new_data_df['FLAG_OWN_REALTY'] = int(self.FLAG_OWN_REALTY)
        new_data_df['FLAG_OWN_CAR'] = int(self.FLAG_OWN_CAR)
        new_data_df['OWN_CAR_AGE'] = float(self.OWN_CAR_AGE)
        new_data_df['NAME_INCOME_TYPE_Working'] = bool(self.NAME_INCOME_TYPE_Working)
        new_data_df['DAYS_EMPLOYED'] = float(self.DAYS_EMPLOYED)
        new_data_df['AMT_GOODS_PRICE'] = float(self.AMT_GOODS_PRICE)
        new_data_df['AMT_CREDIT_SUM_mean'] = float(self.AMT_CREDIT_SUM_mean)
        new_data_df['EXT_SOURCE_1'] = float(self.EXT_SOURCE_1)
        new_data_df['EXT_SOURCE_2'] = float(self.EXT_SOURCE_2)
        new_data_df['EXT_SOURCE_3'] = float(self.EXT_SOURCE_3)

        new_X = new_data_df.drop(columns=['TARGET', 'y_pred_proba', 'y_pred'], errors='ignore')
        new_y_proba = float(model.predict_proba(new_X)[:, 1][0])
        new_y_pred = int(new_y_proba >= BEST_THRESHOLD)

        new_data_df['y_pred_proba'] = new_y_proba
        new_data_df['y_pred'] = new_y_pred
        return new_data_df

@dataclass
class Client_new_credit:
    SK_ID_CURR: int
    FLAG_OWN_REALTY: int
    FLAG_OWN_CAR: int
    OWN_CAR_AGE: float
    NAME_INCOME_TYPE_Working: bool
    DAYS_EMPLOYED: float
    AMT_GOODS_PRICE: float
    AMT_CREDIT_SUM_mean: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float

def Client_credit_from_data(client_data) -> Client_credit:
    # Forcer tous les types natifs Python pour éviter PydanticSerializationError
    return Client_credit(
        SK_ID_CURR=int(client_data['SK_ID_CURR']),
        FLAG_OWN_REALTY=int(client_data['FLAG_OWN_REALTY']),
        FLAG_OWN_CAR=int(client_data['FLAG_OWN_CAR']),
        OWN_CAR_AGE=float(client_data['OWN_CAR_AGE']),
        NAME_INCOME_TYPE_Working=bool(client_data['NAME_INCOME_TYPE_Working']),
        DAYS_EMPLOYED=float(client_data['DAYS_EMPLOYED']),
        AMT_GOODS_PRICE=float(client_data['AMT_GOODS_PRICE']),
        AMT_CREDIT_SUM_mean=float(client_data['AMT_CREDIT_SUM_mean']),
        EXT_SOURCE_1=float(client_data['EXT_SOURCE_1']),
        EXT_SOURCE_2=float(client_data['EXT_SOURCE_2']),
        EXT_SOURCE_3=float(client_data['EXT_SOURCE_3']),
        PRED_PROBA=float(client_data['y_pred_proba']),
        PRED_TARGET=int(client_data['y_pred']),
        TARGET=bool(client_data['TARGET'])
    )

def to_Client_credit(credit: Client_new_credit) -> Client_credit:
    return Client_credit(
        SK_ID_CURR=int(credit.SK_ID_CURR),
        FLAG_OWN_REALTY=int(credit.FLAG_OWN_REALTY),
        FLAG_OWN_CAR=int(credit.FLAG_OWN_CAR),
        OWN_CAR_AGE=float(credit.OWN_CAR_AGE),
        NAME_INCOME_TYPE_Working=bool(credit.NAME_INCOME_TYPE_Working),
        DAYS_EMPLOYED=float(credit.DAYS_EMPLOYED),
        AMT_GOODS_PRICE=float(credit.AMT_GOODS_PRICE),
        AMT_CREDIT_SUM_mean=float(credit.AMT_CREDIT_SUM_mean),
        EXT_SOURCE_1=float(credit.EXT_SOURCE_1),
        EXT_SOURCE_2=float(credit.EXT_SOURCE_2),
        EXT_SOURCE_3=float(credit.EXT_SOURCE_3)
    )

# Charger le modèle
model = mlflow.sklearn.load_model("mlflow_model")

# Charger les données
print("Chargement des données de test...")
temp_df = pd.read_csv(ZIP_TEST_DATA_FILENAME, sep=',', encoding='utf-8', compression='zip')
if "Unnamed: 0" in temp_df.columns:
    temp_df = temp_df.drop(columns=["Unnamed: 0"])

min_SK_ID_CURR = int(temp_df['SK_ID_CURR'].min())
max_SK_ID_CURR = int(temp_df['SK_ID_CURR'].max())
y = temp_df['TARGET']
X = temp_df.drop(columns=['TARGET'], errors='ignore')

y_pred_proba = model.predict_proba(X)[:, 1]
y_pred = np.where(y_pred_proba >= BEST_THRESHOLD, 1, 0)

merged_data_df = pd.concat([
    temp_df,
    pd.DataFrame(y_pred_proba, columns=['y_pred_proba']),
    pd.DataFrame(y_pred, columns=['y_pred'])
], axis=1)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

del temp_df
del y_pred_proba
del y_pred

# Créer l'API
app = FastAPI(debug=True)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de scoring crédit 🚀"}

@app.get("/get_client/{SK_ID_CURR}")
def get_client_by_ID(SK_ID_CURR: int = Path(ge=min_SK_ID_CURR, le=max_SK_ID_CURR)) -> Client_credit:
    part_data_df = merged_data_df[merged_data_df['SK_ID_CURR'] == SK_ID_CURR]
    if part_data_df.empty:
        raise HTTPException(status_code=404, detail="SK_ID_CURR non trouvé !")
    client_dict = part_data_df.iloc[0].to_dict()
    return Client_credit_from_data(client_dict)

# @app.post("/predict")
# def predict_credit(client: Client_new_credit = Body(...)) -> Client_credit:
#     # Convertir en Client_credit
#     client_credit = to_Client_credit(client)

#     # Préparer les données pour la prédiction
#     new_X_df = pd.DataFrame([{
#         'FLAG_OWN_REALTY': int(client_credit.FLAG_OWN_REALTY),
#         'FLAG_OWN_CAR': int(client_credit.FLAG_OWN_CAR),
#         'OWN_CAR_AGE': float(client_credit.OWN_CAR_AGE),
#         'NAME_INCOME_TYPE_Working': int(client_credit.NAME_INCOME_TYPE_Working),
#         'DAYS_EMPLOYED': float(client_credit.DAYS_EMPLOYED),
#         'AMT_GOODS_PRICE': float(client_credit.AMT_GOODS_PRICE),
#         'AMT_CREDIT_SUM_mean': float(client_credit.AMT_CREDIT_SUM_mean),
#         'EXT_SOURCE_1': float(client_credit.EXT_SOURCE_1),
#         'EXT_SOURCE_2': float(client_credit.EXT_SOURCE_2),
#         'EXT_SOURCE_3': float(client_credit.EXT_SOURCE_3),
#     }])

#     # Prédiction
#     y_proba = float(model.predict_proba(new_X_df)[:, 1][0])
#     y_pred = int(y_proba >= BEST_THRESHOLD)

#     # Remplir les résultats
#     client_credit.PRED_PROBA = y_proba
#     client_credit.PRED_TARGET = y_pred

#     return client_credit

# Requête : juste l'identifiant client
class ClientIDRequest(BaseModel):
    SK_ID_CURR: int

@app.post("/predict")
def predict_credit(client: ClientIDRequest) -> dict:
    # Récupérer les données du client
    client_data = merged_data_df.loc[merged_data_df['SK_ID_CURR'] == client.SK_ID_CURR]
    if client_data.empty:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Sélectionner exactement les features utilisées à l’entraînement
    X_client = client_data[X.columns]

    # Prédiction
    try:
        y_proba = float(model.predict_proba(X_client)[:, 1][0])
        y_pred = int(y_proba >= BEST_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {e}")

    return {
        "SK_ID_CURR": client.SK_ID_CURR,
        "PRED_PROBA": y_proba,
        "PRED_TARGET": y_pred
    }
    
if __name__ == '__main__':
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)