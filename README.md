# US Accidents — Severity & Duration API

FastAPI service serving the two production models:
XGBoost severity classifier + XGBoost duration regressor

## Project structure

```
us-accidents-api/
├── app/
│   ├── main.py            
│   ├── schemas.py          
│   ├── preprocessing.py   
│   └── model_loader.py     
├── saved_models/           
├── tests/
│   └── test_api.py        
├── requirements.txt
├── Dockerfile
├── .dockerignore
```


