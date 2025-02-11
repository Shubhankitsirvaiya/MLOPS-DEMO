# Use official Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib pandas numpy scikit-learn

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
