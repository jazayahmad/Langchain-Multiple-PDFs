# Use the official Python image from Docker Hub
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

# Install the package dependencies in the requirements file.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the ./app directory inside the /code directory.
COPY . /app

# # Expose the port for Streamlit (if needed)
# EXPOSE 8501

# Command to run your Python script or Streamlit app
CMD ["streamlit", "run", "your_script.py"]
