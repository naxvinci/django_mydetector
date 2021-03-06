import base64
import io
import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.backend import tensorflow_backend as backend
from django.conf import settings

def detect(upload_image):
  result_name = upload_image.name
  result_list = []
  result_img =''

  cascade_file_path = settings.CASCADE_FILE_PATH
  model_file_path = settings.MODEL_FILE_PATH
  model = tf.keras.models.load_model(model_file_path)
  image = np.asarray(Image.open(upload_image))

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_gs = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
  # 1) cascade사용하기 위한 CascadeClassifier 생성
  cascade = cv2.CascadeClassifier(cascade_file_path)
  # 2) OpenCV이용해서 얼굴 인식 함수 호출 -> detectMultiScale()
  faces = cascade.detectMultiScale(image_gs, 
                            scaleFactor=1.1,
                              minNeighbors=5,
                            minSize=(64, 64))

  # 3) 얼굴 인식 개수는? 
  if len(faces) > 0:
    count = 1
    for (xpos, ypos, width, height) in faces:
      face_image = image_rgb[ypos:ypos+height, xpos:xpos+width]

      # 4) 64보다 작으면 무시 
      if face_image.shape[0] < 64 or face_image.shape[1] < 64:
        continue
      # 5) 64보다 크다면 64로 resize
      face_image = cv2.resize(face_image, (64,64))

      # 6) 붉은 색 사각형 표시 
      cv2.rectangle(image_rgb, (xpos, ypos), 
                          (xpos+width, ypos+height),
                          (255, 0, 0),
                          thickness=2)
      face_image = np.expand_dims(face_image, axis=0)
      name, result = detect_who(model, face_image)

      # 7) 인식 된 얼굴의 이름 표기 
      cv2.putText(image_rgb, name, 
                        (xpos, ypos+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, 
                        (255, 0 ,0), 2)
      result_list.append(result)
      count = count + 1
  
  # 8) 이미지를 PNG파일로 변환 
  is_success, img_buffer = cv2.imencode(".png", image_rgb)
  if is_success:
    # 이미지 -> 메인 메모리의 바이너리 형태 
    io_buffer = io.BytesIO(img_buffer)
    result_img = base64.b64encode(io_buffer.getvalue()).decode().replace("'", "")

  # 9) tensorflow에서 session이 닫히지 않는 문제
  backend.clear_session()

  return (result_list, result_name, result_img)

def detect_who(model, face_image):
    # 예측
    name = ""
    result = model.predict(face_image) # 배열형식
    result_msg = f"손흥민일 가능성: {result[0][0]*100:.3f}% / 유아인일 가능성: {result[0][1]*100:.3f}%"

    name_number_label = np.argmax(result)
    if name_number_label == 0:
        name = "Son"
    elif name_number_label == 1:
        name = "Ain"
    
    return (name, result_msg)