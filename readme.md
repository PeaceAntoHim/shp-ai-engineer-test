# SHP AI Engineer Test Project - Quick Start Guide

A comprehensive AI engineering project with three main components: customer data analysis, receipt management platform, and vector database system.

## Project Structure

```
shp-ai-engineer-test/
├── cutomers/           # CSV analysis and data processing
├── receipt/            # Receipt management platform (FastAPI)
├── vector_db/          # Vector database implementation
└── README.md          # This file
```


## Setup Requirements

### Prerequisites
Make sure you have Docker and Docker Compose installed:
```shell script
docker --version
docker-compose --version
```


### Download Required Dataset
Download the large CSV file for customer analysis:
```shell script
# Download large customer dataset (2M records)
wget -O cutomers/dataset/customers-2000000.csv "https://drive.google.com/uc?id=1IXQDp8Um3d-o7ysZLxkDyuvFj9gtlxqz&export=download"
```

## Running Each Component

### 1. Customer Data Analysis

```shell script
# Navigate to customers directory
cd cutomers/

# Make sure docker-compose file has execute permissions
chmod +x docker-compose.yml

# Build and run the container
docker-compose up --build

# To run in background (detached mode)
docker-compose up --build -d

# To view logs (if running detached)
docker-compose logs -f

# To stop the container
docker-compose down
```


### 2. Receipt Management Platform

```shell script
# Navigate to receipt directory
cd receipt/

# Make build scripts executable
chmod +x local.build.sh
chmod +x stg.build.sh

# Option 1: Use the local build script
./local.build.sh

# Option 2: Use docker-compose directly
docker-compose up --build

# To run in background
docker-compose up --build -d

# Access the application
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# To stop
docker-compose down
```


### 3. Vector Database

```shell script
# Navigate to vector_db directory
cd vector_db/

# Make sure docker-compose has execute permissions
chmod +x docker-compose.yml

# Build and run
docker-compose up --build

# To run in background
docker-compose up --build -d

# To view logs
docker-compose logs -f

# To stop
docker-compose down
```


## One-Line Execution

For each component, you can run everything in one command:

```shell script
# Customer Analysis
cd cutomers && chmod +x docker-compose.yml && docker-compose up --build

# Receipt Platform
cd receipt && chmod +x *.sh && docker-compose up --build

# Vector Database  
cd vector_db && chmod +x docker-compose.yml && docker-compose up --build
```


## Quick Commands Reference

### Build and Run All Components
```shell script
# From project root, run each in separate terminal windows:

# Terminal 1 - Customer Analysis
cd cutomers && docker-compose up --build

# Terminal 2 - Receipt Platform  
cd receipt && docker-compose up --build

# Terminal 3 - Vector Database
cd vector_db && docker-compose up --build
```


### Clean Up Everything
```shell script
# Stop all containers and remove volumes
cd cutomers && docker-compose down -v
cd ../receipt && docker-compose down -v  
cd ../vector_db && docker-compose down -v

# Remove unused Docker resources
docker system prune -f
```


### View Container Status
```shell script
# See all running containers
docker ps

# See all containers (including stopped)
docker ps -a

# View logs for specific container
docker logs customer-app
docker logs ai-engineering-platform
docker logs vector-database
```


## Expected Behavior

- **Customer Analysis**: Processes CSV files and generates analysis reports, then exits
- **Receipt Platform**: Starts a web server on port 8000 with API endpoints, stays running
- **Vector Database**: Runs vector operations and generates performance reports, then exits

## Component Overview

### Customer Data Analysis (`cutomers/`)
- High-performance CSV data processing
- Memory-efficient handling of large datasets (100K and 2M+ records)
- Comprehensive statistical analysis and reporting

### Receipt Management Platform (`receipt/`)
- AI-powered FastAPI platform for receipt processing
- OCR capabilities with Tesseract
- RESTful API with health monitoring
- Access at: http://localhost:8000

### Vector Database (`vector_db/`)
- Custom vector database implementation
- Similarity search capabilities
- Dynamic content generation and performance monitoring

## Troubleshooting

### Permission Issues
```shell script
# Make everything executable recursively
chmod -R +x .

# Or specifically for each directory
chmod -R +x cutomers/
chmod -R +x receipt/ 
chmod -R +x vector_db/
```


### Docker Build Issues
```shell script
# Try building without cache
docker-compose build --no-cache
docker-compose up

# Clean Docker system
docker system prune -af
```


### Port Conflicts
If port 8000 is already in use for the receipt platform:
```shell script
# Check what's using the port
lsof -i :8000

# Kill the process or modify docker-compose.yml to use different port
```


## Verification

### Customer Analysis
- Check `cutomers/output_logs/` for generated analysis reports
- Container should exit with status 0 after completion

### Receipt Platform
#### LOCAL
- Visit http://localhost:8000 to see the API
- Visit http://localhost:8000/docs for interactive API documentation
- Check http://localhost:8000/api/v1/health for health status

#### STG
- Visit http://34.101.61.31:8000/ to see the API
- Visit http://34.101.61.31:8000/docs for interactive API documentation
- Check http://34.101.61.31:8000/api/v1/health for health status

### Vector Database
- Check `vector_db/output_logs/` for performance reports
- Container should exit with status 0 after completion

---

**Quick Start Summary:**
1. Download required dataset using the provided link
2. Navigate to each component directory
3. Run `chmod +x docker-compose.yml` 
4. Run `docker-compose up --build`
5. Access receipt platform at http://localhost:8000
6. Check output logs for results