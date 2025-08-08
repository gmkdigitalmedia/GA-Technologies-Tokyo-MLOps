# 🚀 GP MLOps Platform - Quick Start Guide

## **Prerequisites**

### **Required Software**
- **Docker Desktop** (Windows/Mac) or Docker Engine (Linux)
- **Docker Compose** v2.0+
- **Python 3.8+** 
- **Git**
- **8GB+ RAM available**
- **20GB+ disk space**

### **Check Prerequisites**
```bash
# Check Docker
docker --version
docker-compose --version

# Check Python
python3 --version
pip3 --version

# Check available ports (should be free)
netstat -an | grep -E "2222|2223|2224|2225|2226|2227|2228|2229|2230|2231|2233|2235|2237"
```

---



## **Full MLOps Stack (Advanced)**

### **Step 1: Prepare Environment**
```bash
cd /mnt/c/Users/ibm/Documents/GA

# Create necessary directories
mkdir -p data/models data/artifacts logs monitoring/config airbyte/temporal/dynamicconfig

# Set permissions for scripts
chmod +x start_full_mlops.sh
chmod +x dashboard/start_dashboard.sh
```

### **Step 2: Configure Environment Variables**
Create `.env` file:
```bash
cat > .env << 'EOF'
# Optional - for full functionality
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_USER=your-user
SNOWFLAKE_PASSWORD=your-password
SNOWFLAKE_DATABASE=GP_MLOPS_DW
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
EOF
```

### **Step 3: Start Full MLOps Stack**
```bash
# This will take 5-10 minutes first time (downloading Docker images)
./start_full_mlops.sh
```

### **Step 4: Verify Services**
Wait for all services to start, then check:
- **API Health**: http://localhost:2223/health
- **API Docs**: http://localhost:2223/docs
- **MLflow**: http://localhost:2226
- **Dify**: http://localhost:2230
- **Airbyte**: http://localhost:2237
- **Grafana**: http://localhost:2228 (admin/admin)

---

## **Option 3: Manual Component Start**

### **Backend Only (Minimal)**
```bash
cd dashboard
python3 backend.py
# Access API at http://localhost:2233
```

### **Frontend Only**
```bash
cd dashboard
python3 frontend.py
# Access dashboard at http://localhost:2222
```

### **Docker Services (Individual)**
```bash
# Start specific services
docker-compose up -d postgres redis
docker-compose up -d mlflow
docker-compose up -d grafana prometheus
```

---

## **Troubleshooting**

### **Port Already in Use**
```bash
# Find process using port (example for 2222)
lsof -i :2222  # Mac/Linux
netstat -ano | findstr :2222  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows
```

### **Docker Issues**
```bash
# Reset Docker
docker-compose down -v
docker system prune -a

# Restart Docker Desktop/Service
# Windows: Restart Docker Desktop from system tray
# Linux: sudo systemctl restart docker
```

### **Python Module Not Found**
```bash
# Install missing modules
pip3 install fastapi uvicorn pandas numpy scikit-learn
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Permission Denied**
```bash
# Fix script permissions
chmod +x start_full_mlops.sh
chmod +x dashboard/start_dashboard.sh
chmod +x dashboard/backend.py
chmod +x dashboard/frontend.py
```

### **Database Connection Issues**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart database
docker-compose restart postgres
```

---

## **Quick Test Commands**

### **Test API**
```bash
# Check API health
curl http://localhost:2223/health

# Get MLOps status
curl http://localhost:2223/api/v1/mlops/status

# Test Tokyo dashboard backend
curl http://localhost:2233/
```

### **Test ML Prediction**
```bash
curl -X POST http://localhost:2233/predict/ad-targeting \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "annual_income": 12000000,
    "occupation": "manager",
    "location_preference": "Shibuya",
    "family_size": 2,
    "budget_range": "High"
  }'
```

---

## **Stop Everything**

### **Stop Dashboard**
```bash
# Press Ctrl+C in terminal running dashboard
# Or find and kill processes
pkill -f "python3 backend.py"
pkill -f "python3 frontend.py"
```

### **Stop Docker Services**
```bash
cd /mnt/c/Users/ibm/Documents/GA
docker-compose down

# Remove volumes (full cleanup)
docker-compose down -v
```

---

## **Common Issues & Solutions**

### **Issue 1: "Cannot connect to backend"**
**Solution**: Ensure backend is running on port 2233
```bash
ps aux | grep backend.py
# If not running:
cd dashboard && python3 backend.py
```

### **Issue 2: "Module not found: sklearn"**
**Solution**: Install scikit-learn
```bash
pip3 install scikit-learn==1.3.2
```

### **Issue 3: "Docker daemon not running"**
**Solution**: Start Docker
```bash
# Windows: Start Docker Desktop from Start Menu
# Mac: Start Docker from Applications
# Linux: sudo systemctl start docker
```

### **Issue 4: "Address already in use"**
**Solution**: Kill existing process or change port
```bash
# Find and kill process on port 2222
lsof -ti:2222 | xargs kill -9

# Or modify port in files:
# - dashboard/backend.py (change 2233)
# - dashboard/frontend.py (change 2222)
# - dashboard/index.html (change API_BASE)
```

---

## **Minimal Working Setup**

If you just want to see something working quickly:

```bash
# 1. Navigate to dashboard
cd /mnt/c/Users/ibm/Documents/GA/dashboard

# 2. Install minimal requirements
pip3 install fastapi uvicorn pandas numpy scikit-learn

# 3. Start backend
python3 backend.py &

# 4. Start frontend
python3 frontend.py

# 5. Open browser to http://localhost:2222
```

---

## **Health Check URLs**

Once running, verify these endpoints:

| Service | URL | Expected Response |
|---------|-----|-------------------|
| Tokyo Dashboard | http://localhost:2222 | Dashboard UI |
| Backend API | http://localhost:2233 | JSON response |
| API Docs | http://localhost:2223/docs | Swagger UI |
| MLflow | http://localhost:2226 | MLflow UI |
| Dify | http://localhost:2230 | Dify Console |
| Airbyte | http://localhost:2237 | Airbyte UI |
| Grafana | http://localhost:2228 | Login page |
| Prometheus | http://localhost:2227 | Prometheus UI |

---

## **Support**

If you encounter issues:

1. Check the logs:
   ```bash
   # Dashboard logs
   cat dashboard/logs/backend.log
   
   # Docker logs
   docker-compose logs -f
   ```

2. Verify prerequisites are installed
3. Ensure ports 2222-2237 are free
4. Try the minimal setup first
5. Restart Docker if needed

---

**Ready to go! Start with Option 1 (Quick Start) for fastest results.** 🚀