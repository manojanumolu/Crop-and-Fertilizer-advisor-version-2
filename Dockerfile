# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /code

# Copy your requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your other files
COPY . .

# Start the server (Hugging Face REQUIRES port 7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
