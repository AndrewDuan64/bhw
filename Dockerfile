# Use the official Python image.
FROM python:3.10-slim

# Set a working directory in the container.
WORKDIR /app

# Copy requirements first for better build caching.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files into the container.
COPY . .

# Expose the port Streamlit will run on.
EXPOSE 8501

# Command to run your Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
