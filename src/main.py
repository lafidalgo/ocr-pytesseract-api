from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import pytesseract
import numpy as np
from io import BytesIO
import cv2
import base64

app = FastAPI()

class Params(BaseModel):
    output_type: Optional[str] = "string"
    lang: Optional[str] = "eng"
    config: Optional[str] = "--psm 6"
    nice: Optional[int] = 0 
    timeout: Optional[int] = 0

@app.get("/")
def home():
    return "OCR Pytesseract with FastAPI - Version 1.0"

@app.post("/ocr/")
async def submit(params: Params = Depends(), files: List[UploadFile] = File(...)):
    results = {}

    for file in files:
        # Read the image file as bytes
        img_data = await file.read()

        # Convert the image bytes to a PIL Image
        img = Image.open(BytesIO(img_data))

        # Convert the PIL Image to an OpenCV image (numpy array)
        img_cv2 = np.array(img)

        # Convert the image to grayscale (1 channel)
        if len(img_cv2.shape) >= 3:
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

        # Up-sample
        img_cv2 = cv2.resize(img_cv2, (0, 0), fx=2, fy=2)

        """
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img_cv2 = cv2.dilate(img_cv2, kernel, iterations=1)
        img_cv2 = cv2.erode(img_cv2, kernel, iterations=1)

        # Apply blur to smooth out the edges
        img_cv2 = cv2.GaussianBlur(img_cv2, (5, 5), 0)
        """
        # https://stackoverflow.com/questions/71289347/pytesseract-improving-ocr-accuracy-for-blurred-numbers-on-an-image
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_cv2 = cv2.filter2D(img_cv2, -1, sharpen_kernel)
        img_cv2 = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        """
        # Apply threshold to get image with only b&w (binarization)
        img_cv2 = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Invert the image
        img_cv2 = cv2.bitwise_not(img_cv2)
        """

        # Down-sample
        img_cv2 = cv2.resize(img_cv2, (0, 0), fx=0.5, fy=0.5)

        # Apply tesseract
        ocr = pytesseract.image_to_string(img_cv2, lang=params.lang,
                                        config=params.config,
                                        output_type=params.output_type,
                                        nice=params.nice,
                                        timeout=params.timeout)
        
        _, img_bytes = cv2.imencode('.jpg', img_cv2)
        final_image_base64 = base64.b64encode(img_bytes).decode('utf-8')

        results[file.filename] = {}
        results[file.filename]['ocr'] = ocr
        results[file.filename]['final_image_base64'] = final_image_base64

    return {"results": results,
            "params": params}