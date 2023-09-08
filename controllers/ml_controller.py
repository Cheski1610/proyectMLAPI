from fastapi import APIRouter, HTTPException, status
from models import Prediction_Input
from models import Prediction_Output

import pickle
import pandas as pd

with open('modelo_logist.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

##print(modelo_cargado)

router = APIRouter()

preds = []

@router.get("/ml")
def get_preds():    
    return preds

@router.post("/ml",status_code=status.HTTP_201_CREATED, response_model=Prediction_Output)
def predict(pred_input: Prediction_Input):
     argument_values = [pred_input.text_input.split(',')]
     datos_nuevos = pd.DataFrame(argument_values, columns=["EK", "Skewness"])
     prediction_f = modelo_cargado.predict(datos_nuevos)
     prediction_dict = {"id": str(pred_input.id),"text_input":str(pred_input.text_input),"pred": float(prediction_f[0])}
     preds.append(prediction_dict)

     return prediction_dict

@router.put('/ml/{pred_input.id}')
def update_predict(pred_input: Prediction_Input):    
    for pred_item in preds:        
        if pred_item["id"] == pred_input.id:            
            pred_item["text_input"] = pred_input.text_input
            argument_values = [pred_input.text_input.split(',')]
            datos_nuevos = pd.DataFrame(argument_values, columns=["EK", "Skewness"])
            prediction_f = modelo_cargado.predict(datos_nuevos)            
            pred_item["pred"] = float(prediction_f[0])            
            return {"message" : "Actualizado correctamente"}    
        raise HTTPException(status_code=404,detail="Task no encontrada")
    

@router.delete('/ml/{pred_input.id}')
def delete_task(pred_input: Prediction_Input):    
    for pred_item in preds:        
        if pred_item["id"] == pred_input.id:            
            preds.remove(pred_item)            
            return {"message" : "Eliminado correctamente"}    
    raise HTTPException(status_code=404,detail="Task no encontrada")