#MODELO ARCHIVOS
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from datetime import datetime
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
#descarga datos
from roboflow import Roboflow
rf = Roboflow(api_key="2KfXpvSUP4FkCcSGa70u")
project = rf.workspace("piit").project("screw-bolt-counter")
dataset = project.version(1).download("coco-segmentation")

#/////////////////////////////////////////////////configuraciones 1

DATA_SET_NAME = dataset.name.replace(" ", "-")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
#CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
CONFIG_FILE_PATH = f"./detectron2/configs/COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 2000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 2

# OUTPUT DIR
OUTPUT_DIR_PATH = os.path.join(
    DATA_SET_NAME,
    ARCHITECTURE,
    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
)

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

#//////////////////////////////////Configuraciones 2
cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(CONFIG_FILE_PATH)
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('')
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH

#///////////////////////////Configuraciones 3
cfg.MODEL.WEIGHTS = './model_final.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

#////////////////////////////configuraciones 4
# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "train", ANNOTATIONS_FILE_NAME)

#///////////////////////////////Hacemos el registro
register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

#////////////////////////////////////////Configuracion 5
metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)




#//////////////////////////////API DE FLASK


import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename



#: MODELO______________
def procesador(imgp):
    img = cv2.imread(imgp)
    outputs = predictor(img)
    visualizer = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW
    )
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Generar un nombre único para la imagen procesada
    unique_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    
    # Ruta completa para guardar la imagen procesada en la carpeta 'uploads'
    processed_img_path = os.path.join("static", "uploads", unique_name)
    print(processed_img_path)

        # Obtener las instancias detectadas
    instances = outputs["instances"]
    
    # Obtener la cantidad de detecciones
    num_detections = len(instances)
    
    # Mostrar la cantidad de detecciones
    print("Cantidad de detecciones:", num_detections)
    # Guardar la imagen procesada
    cv2.imwrite(processed_img_path, out.get_image()[:, :, ::-1])

    return unique_name, num_detections




app = Flask(__name__, static_url_path='/static')

# Configuración para el manejo de archivos
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Verificar si se envió un archivo
        if 'archivo' not in request.files:
            return redirect(request.url)
        file = request.files['archivo']
        # Si el usuario no seleccionó un archivo, se redirige a la página de inicio
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join("static", "uploads", filename))
            img_path = os.path.join("static", "uploads", filename)

            # Procesar la imagen y obtener el nombre del archivo procesado y el num_detections
            processed_img_name, num_detections = procesador(img_path)

            # Ruta relativa de la imagen original (desde 'static')
            original_img = url_for('static', filename=os.path.join('uploads', filename))

            # Ruta relativa de la imagen procesada (desde 'static')
            processed_img = url_for('static', filename=os.path.join('uploads', processed_img_name))

            # Renderizar la plantilla 'index.html' y pasar las rutas de las imágenes y num_detections
            return render_template('index.html', original_img=original_img, processed_img=processed_img, num_detections=num_detections)

    # Renderizar la plantilla 'index.html' sin rutas de imágenes y num_detections
    return render_template('index.html', original_img=None, processed_img=None, num_detections=None)



if __name__ == '__main__':
    app.run()
