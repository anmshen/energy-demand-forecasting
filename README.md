## 📌 Project Overview

This project implements a **multi-modal CNN–Transformer architecture** for **day-ahead electricity demand forecasting** across ISO New England load zones. The model fuses **high-dimensional spatial weather data** with **temporal tabular signals** (energy demand and calendar features) to predict the next 24 hours of energy consumption.

Electricity demand is strongly influenced by weather patterns and human activity cycles. To capture these dependencies, the model jointly learns:

* **Spatial features** from large-scale weather maps
* **Temporal dynamics** from historical demand and calendar signals
* **Cross-modal interactions** between weather and energy usage

---

## 🧠 Model Architecture

The core model is an **encoder–decoder CNN-Transformer** designed for efficient spatiotemporal learning:

### 1. Spatial Tokenization (CNN)

Weather maps (450×449×7) are processed using a shared CNN to produce a compact grid of **spatial tokens**, preserving geographic structure while reducing dimensionality.

### 2. Tabular Tokenization (MLP)

* Historical demand + calendar features → embedded into **tabular tokens**
* Future inputs use **calendar features only**, with demand zero-padded

### 3. Sequence Construction

For each timestep:

* Spatial tokens and a tabular token are combined
* Spatial and temporal positional embeddings are added
* All timesteps are flattened into a unified sequence

### 4. Transformer Encoder–Decoder

* **Encoder:** Processes historical sequences using self-attention
* **Decoder:** Processes future sequences using:

  * Self-attention (within future horizon)
  * Cross-attention (to historical encoder memory)

This design enforces a **causal structure**, where future predictions depend only on past information.

### 5. Prediction Head

Decoder outputs are reshaped and passed through an MLP to generate **24-hour forecasts for all load zones**, followed by de-normalization to real-world energy values.

---

## 📊 Problem Setup

The model takes as input:

* **Historical (168 hours):**

  * Weather maps
  * Energy demand per zone
  * Calendar features
* **Future (24 hours):**

  * Forecasted weather
  * Calendar features

It outputs:

* **24-hour ahead predictions** of energy demand for all zones

Performance is evaluated using **Mean Absolute Percentage Error (MAPE)**, a standard metric in the energy industry. 

---

## 🚀 Key Features

* **Multi-modal fusion** of spatial and tabular data
* **Token-based representation** of weather grids (CNN → Transformer)
* **Efficient encoder–decoder design** reducing attention complexity
* **Causality-aware forecasting** via cross-attention
* Scalable to large spatiotemporal datasets

---

## 🎯 Goal

The objective is to build a **state-of-the-art forecasting model** that captures both:

* Large-scale **weather-driven demand patterns**
* Fine-grained **temporal consumption behavior**

This approach mirrors real-world energy forecasting systems used in power grid operations. 

## File Locations
* folder part1_CNNTransformerEncoder implements a CNN-Encoder model for energy prediction. The code used to generate attention maps can also be found in this folder.
* folder part2_CNNEncoderDecoder implements a CNN-Encoder-Decoder model for energy prediction.
