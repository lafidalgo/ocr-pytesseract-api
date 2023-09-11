# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for pytesseract
RUN apt-get update && apt-get install -y libtesseract-dev tesseract-ocr

# Install language traineddata packages for english and portuguese
RUN apt-get install -y tesseract-ocr-eng tesseract-ocr-por

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run uvicorn to start the FastAPI application
CMD ["uvicorn", "src.apis:app", "--host", "0.0.0.0", "--port", "8000"]