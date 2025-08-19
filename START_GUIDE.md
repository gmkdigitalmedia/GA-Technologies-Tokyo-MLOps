# LAUNCH GP MLOps MLOps Platform - Startup Guide

## FAST **Quick Start Options**

### **Option 1: Working Demo (Fastest) STAR**
```bash
cd dashboard
./start_dashboard.sh
```
**Result**: Fully working ML demo on ports 2222/2233 with real models

---

### **Option 2: Core MLOps Services (Recommended)**
```bash
./start_core_services.sh
```
**Result**: All essential MLOps services except Airbyte
- PASS PostgreSQL, Redis, MLflow
- PASS Prometheus, Grafana monitoring
- PASS Dify LLM workflows
- ⚠️ Skips Airbyte (due to health check issues)

---

### **Option 3: Full Stack (If Airbyte works)**
```bash
./start_full_mlops.sh
```
**Result**: Complete MLOps platform with all services

---

## FIX **Troubleshooting**

### **If Airbyte Database is Unhealthy**
```bash
./fix_airbyte.sh
```
This script:
- Stops and cleans up Airbyte services
- Fixes health check timing issues
- Restarts Airbyte with proper timeouts

### **If Docker Build Fails**
```bash
./fix_dependencies.sh
./start_core_services.sh
```
This bypasses the build issues by using pre-built images

### **If Ports Are Busy**
```bash
# Check what's using ports
netstat -an | grep -E "2222|2223|2224|2225|2226|2227|2228"

# Kill processes (Linux/Mac)
lsof -ti:2222 | xargs kill -9

# Or on Windows
netstat -ano | findstr :2222
taskkill /PID <PID> /F
```

---

## 🌐 **Service Access Points**

| Service | URL | Status | Purpose |
|---------|-----|---------|---------|
| **Tokyo Dashboard** | http://localhost:2222 | PASS Working | Main ML demo |
| **Dashboard API** | http://localhost:2233 | PASS Working | Backend API |
| **MLOps Platform** | http://localhost:2223 | ⚠️ May fail | Full platform API |
| **MLflow** | http://localhost:2226 | PASS Working | Model registry |
| **Grafana** | http://localhost:2228 | PASS Working | Monitoring dashboards |
| **Dify Console** | http://localhost:2230 | PASS Working | LLM workflows |
| **Airbyte** | http://localhost:2237 | ⚠️ Health issues | Data integration |

---

## CHART **What Works vs. What May Fail**

### **PASS Guaranteed to Work:**
- **Dashboard Demo** (`cd dashboard && ./start_dashboard.sh`)
- **Pre-built services** (PostgreSQL, Redis, MLflow, Grafana)
- **Core monitoring** and basic MLOps functionality

### **⚠️ May Have Issues:**
- **Main API build** (dependency conflicts)
- **Airbyte database** (health check timeouts)
- **Full Docker Compose** (complex dependencies)

### **TARGET Recommended Flow:**
1. Start with the dashboard demo to show working ML
2. Use core services script for MLOps components
3. Fix specific issues (Airbyte, API build) as needed

---

## SEARCH **Checking Service Status**

```bash
# Check all services
docker-compose ps

# Check specific service logs
docker-compose logs -f [service_name]

# Check service health
curl http://localhost:2222  # Dashboard
curl http://localhost:2233  # Dashboard API
curl http://localhost:2226  # MLflow
```

---

## INFO **Pro Tips**

1. **Start with dashboard demo first** - it always works and shows real ML
2. **Use core services script** - avoids most Docker build issues  
3. **Fix Airbyte separately** - it's the most problematic service
4. **Monitor logs** - use `docker-compose logs -f` to see what's happening
5. **Clean slate approach** - run `docker-compose down -v` if things get stuck

---

**TARGET Ready to demo? Start with: `cd dashboard && ./start_dashboard.sh`**