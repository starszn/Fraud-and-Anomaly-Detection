# 🔍 Fraud Detection Dashboard

A multi‑page, interactive fraud‑analytics dashboard built with **Python** and **Streamlit**, designed to simulate the workflow of a real fraud‑operations team.  
The project integrates machine‑learning anomaly detection with modern UI design to deliver a polished, production‑style experience.

---

## 🚀 Features

### **📊 System Overview**
- Key fraud metrics (total transactions, anomalies, unique customers)
- Donut chart showing anomaly score distribution
- Highest‑risk customers table with risk‑tier badges

### **🔎 Customer Search & Investigation**
- Search customers by ID
- View transaction history, anomaly scores, and risk level
- Glass‑styled tables for clean, modern readability

### **🔥 Risk Ranking**
- Global ranking of customers by anomaly score
- High/Medium/Low risk badges for fast triage

### **🖥️ Device & IP Risk Panel**
- Highest‑risk devices and IP addresses
- Shared‑device detection (fraud ring indicator)
- Auto‑generated device/IP fields for simulation

### **🚨 Fraud Alerts Feed (Realism Upgrade)**
- Real‑time‑style feed of high‑risk transactions
- Adjustable anomaly threshold and alert count
- Ideal for simulating live fraud monitoring

---

## 🧠 Machine Learning Component

The dashboard integrates an **Isolation Forest** model to:
- Score transactions for anomaly likelihood  
- Flag potential fraud  
- Support downstream risk ranking and alerting  

Model outputs are cached for fast UI performance.

---

## 🎨 UI/UX Design

The interface uses a custom **glass‑morphism aesthetic**, including:
- Layered gradient background blobs  
- Frosted‑glass cards  
- Glass‑styled HTML tables  
- Smooth hover animations  
- Transparent Altair charts  

This creates a polished, modern, fintech‑style experience.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Altair / Plotly**
- **Isolation Forest (scikit‑learn)**
- **Custom CSS (glassmorphism, gradients, animations)**

---

---

## 🎯 Purpose

This project was built to demonstrate:
- Fraud analytics concepts  
- Machine‑learning integration  
- Dashboard engineering  
- UI/UX design  
- Real‑world investigative workflows  
## 📁 Website

https://fraudanomalydetection.streamlit.app/

