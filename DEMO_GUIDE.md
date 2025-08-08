# 🏢 GP MLOps - MLOps Platform Demo Guide

## 📁 Demo Structure Overview

This repository contains **three different demo types** for different audiences:

### 1. 🚀 **Main Dashboard** (`/dashboard/`)
**Best for: Live demos, technical presentations, quick showcase**
- **Primary Interface**: Tokyo Real Estate A/B Testing Dashboard
- **Ports**: Frontend (2222), Backend (2233)
- **Start**: `cd dashboard && ./start_dashboard.sh`
- **Features**: Real-time ML models, ad serving, A/B testing

### 2. 🏢 **Enterprise Demo** (`/enterprise_demo/`)
**Best for: Executive presentations, business stakeholders, ROI discussions**
- **Focus**: Business impact, financial metrics, strategic value
- **Audience**: C-level executives, board members, investors
- **Features**: Revenue dashboards, market analysis, competitive positioning

### 3. 🔧 **Technical Demo** (`/technical_demo/`)
**Best for: Technical teams, developers, architects**
- **Focus**: MLOps pipeline, model development, system architecture
- **Audience**: Engineering teams, technical stakeholders, developers
- **Features**: Model training, API testing, technical deep-dives

---

##  Quick Start by Audience


### **For General Audience → Main Dashboard**
```bash
cd /mnt/c/Users/ibm/Documents/GA/dashboard
./start_dashboard.sh
```
**Shows**: Working ML models, real-time analytics, interactive features

### **For Technical Teams → Full MLOps Stack**
```bash
cd /mnt/c/Users/ibm/Documents/GA
./start_full_mlops.sh
```
**Shows**: Complete MLOps pipeline, model deployment, technical architecture

---

## 🌐 Access Points Summary

| Demo Type | Primary URL | Secondary | Purpose |
|-----------|-------------|-----------|---------|
| **Main Dashboard** | http://localhost:2222 | http://localhost:2233/docs | Live ML demo |
| **Enterprise** | http://localhost:3001 | - | Executive presentation |
| **Technical** | http://localhost:3002 | - | Technical deep-dive |
| **Full MLOps** | http://localhost:2223 | Multiple services | Complete platform |

---

## 📊 Service Ports Reference

### **Core Services**
- **2222**: Main Dashboard (Frontend)
- **2223**: MLOps Platform API
- **2224**: PostgreSQL (Main)
- **2225**: Redis
- **2226**: MLflow Model Registry
- **2227**: Prometheus Monitoring
- **2228**: Grafana Dashboards

### **LLM & Data Services**
- **2229**: Dify API (LLM Workflows)
- **2230**: Dify Web Console
- **2233**: Tokyo Dashboard Backend
- **2234**: PostgreSQL (Dify)
- **2235**: PostgreSQL (Airbyte)
- **2236**: Airbyte API
- **2237**: Airbyte Web Interface

---

##  Demo Scenarios by Use Case

### **Scenario 1: "Show me what it does"**
**→ Use Main Dashboard**
1. Start: `cd dashboard && ./start_dashboard.sh`
2. Open: http://localhost:2222
3. Demo: Live ML predictions, real-time metrics, interactive features



### **Scenario 3: "How does it work technically?"**
**→ Use Full MLOps Stack**
1. Start: `./start_full_mlops.sh`
2. Access: Multiple technical interfaces
3. Show: MLflow, Dify, Airbyte, Prometheus, Grafana

### **Scenario 4: "Can I test the APIs?"**
**→ Use Technical Demo**
1. Start: `cd technical_demo && ./run_real_demo.sh`
2. Use: API testing scripts, model training examples
3. Show: Direct API interactions, model development workflow

---

## 🔧 Troubleshooting

### **Port Conflicts**
If ports are in use:
```bash
# Check which ports are busy
netstat -an | grep -E "2222|2223|2224|2225|2226|2227|2228|2229|2230|2233|2235|2237"

# Kill processes on specific ports
lsof -ti:2222 | xargs kill -9  # Linux/Mac
netstat -ano | findstr :2222  # Windows (then taskkill /PID <PID> /F)
```

### **Service Not Starting**
```bash
# Check Docker status
docker ps

# Restart specific services
docker-compose restart [service_name]

# View logs
docker-compose logs -f [service_name]
```

### **Database Issues**
```bash
# Reset databases
docker-compose down -v
docker-compose up -d postgres redis
```

---

## 📚 Documentation Files

### **Main Documentation**
- `README.md` - Overall platform overview
- `QUICK_START_GUIDE.md` - Getting started guide
- `MLOPS_ARCHITECTURE.md` - Technical architecture
- `DEMO_GUIDE.md` - This file (demo organization)

### **Demo-Specific READMEs**
- `dashboard/README.md` - Main dashboard features
- `technical_demo/README.md` - Technical implementation details

---

## 🎯 **Recommendation by Audience**

### **First-time Visitors**
Start with: **Main Dashboard** (`/dashboard/`)
- Easiest to set up and run
- Shows immediate value with working ML
- Interactive and engaging


### **Technical Teams**
Deploy: **Full MLOps Stack** (`./start_full_mlops.sh`)
- Complete technical architecture
- All MLOps components running
- Production-like environment

---

**Ready to start? Choose your demo type above and follow the quick start commands!**