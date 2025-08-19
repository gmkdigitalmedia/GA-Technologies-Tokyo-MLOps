# 🧪 Async Functionality Testing Guide

This directory contains comprehensive async functionality tests for the GP MLOps Platform.

## LAUNCH Quick Start - Run Complete Test

```bash
# Make the script executable (if not already)
chmod +x test_async_complete.sh

# Run the complete async verification
./test_async_complete.sh
```

This script will:
1. PASS Check system requirements
2. 🐳 Start Docker services if needed  
3. 🧪 Run comprehensive async tests
4. CHART Generate detailed reports

## SECTION Available Test Files

### 1. `test_async_complete.sh` STAR **RECOMMENDED**
**Complete test suite that handles everything automatically**
- Checks if Docker services are running
- Starts services if needed
- Runs tests inside Docker container with all dependencies
- Falls back to external tests if needed
- Generates comprehensive reports

### 2. `test_async_docker.py`
**Full system test (requires running Docker services)**
```bash
# Start services first
docker-compose up -d

# Run test (inside container or with aiohttp installed)
python3 test_async_docker.py
```

### 3. `run_async_test.py`
**Standalone test (works without dependencies)**
```bash
python3 run_async_test.py
```

### 4. `test_async_simple.py`
**Code analysis test (no services required)**
```bash
python3 test_async_simple.py
```

## TARGET What Gets Tested

### PASS Core Async Functionality
- Basic async/await syntax
- Concurrent execution
- Async context managers
- Error handling in async context

### 🌐 API Endpoint Testing
- FastAPI async routes
- Response times and concurrency
- Error handling (404, validation errors)
- Health check endpoints

### WAIT Service Integration
- Health → Prediction workflows
- Parallel service calls
- MLOps pipeline status
- End-to-end async workflows

### FAST Performance Testing
- Concurrent request handling (5, 10, 20 requests)
- Response time benchmarks
- Requests per second (RPS) metrics
- Sequential vs parallel execution speedup

## CHART Test Results

After running tests, you'll get:

1. **Console Output**: Real-time test progress and results
2. **JSON Report**: Detailed results in `full_async_verification_report.json`
3. **Assessment**: Overall async health rating

### Sample Output:
```
TARGET Assessment: STAR EXCELLENT - All async functionality working perfectly!
CHART Tests: 15/16 passed
PASS Success Rate: 93.8%
```

## 🐳 Docker Service Requirements

For full testing, these services should be running:

```bash
# Start core services
docker-compose up -d api postgres redis

# Or start all services
docker-compose up -d
```

**Service URLs:**
- Main API: http://localhost:2223
- Health Check: http://localhost:2223/health/
- MLflow: http://localhost:2226
- Prometheus: http://localhost:2227

## FIX Troubleshooting

### Services Not Starting
```bash
# Check Docker status
docker-compose ps

# Check logs
docker-compose logs api

# Restart services
docker-compose down && docker-compose up -d
```

### Dependencies Missing
The complete test script handles this automatically, but if running individual tests:

```bash
# For external testing (not in container)
pip install aiohttp

# Or use the container-based tests
docker-compose exec api python test_async_docker.py
```

### Port Conflicts
If ports 2223-2230 are in use, update `docker-compose.yml` port mappings.

## UP Performance Benchmarks

**Expected Results for Healthy System:**
- Basic async execution: < 0.01s
- Concurrent requests (10): > 2x speedup vs sequential
- API response time: < 0.5s
- Health check: < 0.1s
- Success rate: > 90%

## TOOL Advanced Usage

### Custom Base URL
```bash
python3 test_async_docker.py http://your-custom-url:port
```

### Run Specific Test Categories
Edit the test files to comment out unwanted test sections.

### Integration with CI/CD
```bash
# Exit code 0 = success, 1 = failure
./test_async_complete.sh
echo "Exit code: $?"
```

---

## TARGET Summary

The async functionality testing suite provides comprehensive verification that:

1. PASS **Basic async patterns work correctly**
2. PASS **FastAPI handles concurrent requests efficiently** 
3. PASS **Service integrations use async properly**
4. PASS **Error handling works in async contexts**
5. PASS **Performance meets expectations**

Run `./test_async_complete.sh` to verify your entire async implementation! LAUNCH