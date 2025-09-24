# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
# This includes main.py, the /model folder, and the /artifacts folder
COPY . .

# 6. Make port 8000 available to the world outside this container
EXPOSE 8000

# 7. Define the command to run your app using uvicorn
# This will start the server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]