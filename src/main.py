import os
import uvicorn
from config import UPLOAD_PATH
from encode import Resnet50
from fastapi import FastAPI
from logs import LOGGER
from operations.load import imgdir2_vectors
from starlette.middleware.cors import CORSMiddleware
import json

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
MODEL = Resnet50()
#MILVUS_CLI = MilvusHelper()

# Mkdir '/tmp/search-images'
if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")


@app.post('/img/imgDir2Vectors')
def imgDir2Vectors(url: str = None):
    try:
        data = imgdir2_vectors(url, MODEL)
        return json.dumps(data)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400

@app.post('/img/img2Vectors')
def img2Vectors(url: str = None):
    try:
        feat = MODEL.resnet50_extract_feat(url).tolist()
        LOGGER.info(f"-----------------------------------------------------")
        json_data = json.dumps(feat)
        return json_data
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5000 )

