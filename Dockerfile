# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Poetry lock file and pyproject.toml to the container
COPY pyproject.toml poetry.lock ./

# Install Poetry and any other system dependencies
RUN pip install poetry==1.8.0 && poetry export -f requirements.txt --output requirements.txt --without-hashes

RUN pip install -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]