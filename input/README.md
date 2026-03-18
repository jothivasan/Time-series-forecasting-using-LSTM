# 📁 INPUT FOLDER - Sample Datasets for Upload

This folder contains sample CSV files that you can use to test the **custom dataset upload** feature of the Multi-Domain LSTM Forecasting System.

---

## 📊 Available Sample Files

### 1. **sample_retail_sales.csv**
- **Domain:** Retail/Sales
- **Type:** Multivariate
- **Features:** Date, Sales, Temperature, Promotion, Holiday
- **Records:** 30 days
- **Use Case:** Retail sales forecasting with external factors

### 2. **sample_weather_data.csv**
- **Domain:** Weather/Climate
- **Type:** Multivariate
- **Features:** Date, Temperature, Humidity, Pressure, WindSpeed
- **Records:** 30 days
- **Use Case:** Weather pattern prediction

### 3. **sample_production_data.csv**
- **Domain:** Manufacturing/Production
- **Type:** Univariate
- **Features:** Date, Production
- **Records:** 24 months
- **Use Case:** Simple production forecasting

### 4. **sample_energy_hourly.csv**
- **Domain:** Energy
- **Type:** Multivariate
- **Features:** DateTime, Consumption, Temperature, DayOfWeek
- **Records:** 24 hours
- **Use Case:** Hourly energy consumption forecasting

### 5. **sample_crypto_prices.csv**
- **Domain:** Finance/Crypto
- **Type:** Multivariate (High Volatility)
- **Features:** Date, Open, High, Low, Close, Volume
- **Records:** 30 days
- **Use Case:** Price trend prediction in volatile markets

### 6. **sample_website_traffic.csv**
- **Domain:** Tech/Analytics
- **Type:** Multivariate
- **Features:** Date, Visits, UniqueUsers, BounceRate, AvgSessionDuration
- **Records:** 30 days
- **Use Case:** Web traffic and user engagement forecasting

### 7. **sample_air_quality.csv**
- **Domain:** Environment
- **Type:** Multivariate
- **Features:** DateTime, PM2.5, PM10, NO2, SO2, AQI
- **Records:** 24 hours
- **Use Case:** Environmental pollution monitoring and forecasting

### 8. **sample_restaurant_bookings.csv**
- **Domain:** Hospitality/Service
- **Type:** Univariate (Seasonal)
- **Features:** Date, Bookings
- **Records:** 30 days
- **Use Case:** Demand planning for staffing and inventory

---

## 🚀 How to Use

### **Method 1: Drag & Drop**
1. Open the dashboard at http://localhost:5001
2. Navigate to **"Domain Selection"** tab
3. Drag any CSV file from this folder and drop it on the upload zone
4. System will automatically analyze the dataset

### **Method 2: Browse & Upload**
1. Open the dashboard at http://localhost:5001
2. Navigate to **"Domain Selection"** tab
3. Click on the upload zone
4. Browse to this `input/` folder
5. Select any CSV file
6. Click Open

---

## 📋 CSV File Requirements

For your own custom datasets, ensure they meet these requirements:

### **Mandatory:**
- ✅ CSV format with headers
- ✅ At least one date/time column (Date, DateTime, Month, etc.)
- ✅ At least one numeric column for forecasting
- ✅ Minimum 50-100 records recommended

### **Optional:**
- Additional numeric features (for multivariate forecasting)
- Categorical features (will be handled automatically)

### **Date Column Formats Supported:**
- `YYYY-MM-DD` (e.g., 2024-01-15)
- `YYYY-MM-DD HH:MM:SS` (e.g., 2024-01-15 14:30:00)
- `MM/DD/YYYY` (e.g., 01/15/2024)
- `Month YYYY` (e.g., January 2024)

---

## 🎯 Testing Workflow

1. **Upload a sample file** from this folder
2. **Review statistics** in Data Analysis tab
3. **Configure preprocessing** (scaling, missing values, etc.)
4. **Set LSTM parameters** in Model Configuration
5. **Train the model** in Training tab
6. **Generate forecasts** in Forecasting tab
7. **Evaluate performance** in Evaluation tab

---

## 💡 Tips for Best Results

### **For Short Datasets (<100 records):**
- Use smaller lookback window (6-8)
- Reduce LSTM units (50-75)
- Fewer epochs (50-100)

### **For Long Datasets (>500 records):**
- Larger lookback window (20-30)
- More LSTM units (150-200)
- More epochs (200-500)

### **For Noisy Data:**
- Enable rolling statistics
- Use robust scaling
- Increase dropout rate (0.3-0.4)

---

## 📝 Creating Your Own Dataset

To create your own CSV file for upload:

```csv
Date,YourTargetVariable,Feature1,Feature2
2024-01-01,100,25.5,1
2024-01-02,105,26.2,0
2024-01-03,98,24.8,1
...
```

**Key Points:**
- First column should be date/time
- Include your target variable (what you want to forecast)
- Add any relevant features that might influence the target
- Ensure data is in chronological order
- No missing headers

---

## 🔍 Troubleshooting

### **Issue: "No file selected" error**
**Solution:** Make sure you're selecting a .csv file

### **Issue: "Error loading dataset"**
**Solution:** 
- Check if file has proper headers
- Ensure date column is in recognized format
- Verify at least one numeric column exists

### **Issue: "Sample file not found"**
**Solution:** 
- Use custom upload instead of predefined domains
- Check file path is correct

---

## 📚 Example Use Cases

### **Retail Sales Forecasting**
Use `sample_retail_sales.csv` to predict:
- Daily sales trends
- Impact of promotions
- Holiday effects
- Temperature influence

### **Weather Prediction**
Use `sample_weather_data.csv` to forecast:
- Temperature patterns
- Humidity trends
- Pressure changes
- Wind speed variations

### **Production Planning**
Use `sample_production_data.csv` to predict:
- Monthly production levels
- Seasonal trends
- Growth patterns

### **Energy Management**
Use `sample_energy_hourly.csv` to forecast:
- Hourly consumption
- Peak demand times
- Temperature impact
- Day-of-week patterns

---

## ✨ Advanced Features

Once you upload a file, you can:
- ✅ **Auto-detect** date and target columns
- ✅ **Create lag features** for better predictions
- ✅ **Generate rolling statistics** (mean, std)
- ✅ **Handle missing values** automatically
- ✅ **Scale data** using multiple methods
- ✅ **Visualize** training progress
- ✅ **Compare** different forecast horizons

---

## 🎓 For Academic Projects

These sample files are perfect for:
- Demonstrating multi-domain capability
- Testing different preprocessing strategies
- Comparing univariate vs multivariate forecasting
- Showcasing system flexibility
- Preparing viva demonstrations

---

**Ready to test? Upload any file from this folder and start forecasting!** 🚀

For more information, see the main README.md in the project root.
