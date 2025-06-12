# ----------------------------- IMPORTS -----------------------------
from fastapi import FastAPI, HTTPException,File, UploadFile, Form, Query, status,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field  # Add the missing import here
import os
import httpx
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
import requests
import re
from google import genai
from PIL import Image
import io
import google.generativeai as genai
import tensorflow as tf
import json
from datetime import datetime, timedelta
from meteostat import Point, Hourly, Monthly, Daily
from typing import List,Literal,Dict,Any
from googletrans import Translator
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv # Import load_dotenv
import logging
import time
load_dotenv()
# ----------------------------- FASTAPI APP INITIALIZATION -----------------------------
app = FastAPI()

# ----------------------------- CORS SETUP -----------------------------
# Allow frontend to call the backend without CORS issues

origins = [
    "http://localhost:5500",  # For local testing (default frontend port)
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
    "https://agrosarthi-frontend.web.app",  # Deployed frontend URL
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Setup logging to a file
logging.basicConfig(filename="api_metrics.log", level=logging.INFO, format="%(asctime)s - %(message)s")

@app.middleware("http")
async def log_api_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000  # in milliseconds

    logging.info(f"{request.method} {request.url.path} took {duration:.2f} ms with status {response.status_code}")
    return response
# ----------------------------- MODEL LOADING -----------------------------
# Load Crop Prediction Model
# Load model and label encoder with specific names
crop_model = tf.keras.models.load_model("model\crop_suggestion_model.h5")
crop_label = joblib.load("model\label_encoder.pkl")

# Load label mapping (optional for readability)
with open("model\label_mapping.json", "r") as f:
    crop_label_mapping = json.load(f)
    crop_label_mapping = {int(k): v for k, v in crop_label_mapping.items()}



# Load Price Estimation Model
price_model_path = "model/crop_price_model.pkl"
try:
    price_model = joblib.load(price_model_path)
except Exception as e:
    print(f"Error loading price estimation model: {e}")
    price_model = None

# Load yeild Prediction Model
yield_model_path = "model/yield_prediction_model.pkl"
try:
    yield_model = joblib.load(yield_model_path)
except Exception as e:
    print(f"Error loading crop prediction model: {e}")
    yeild_model = None

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct full path to CSV inside datasets folder
csv_path = os.path.join(BASE_DIR, 'datasets', 'villages_with_coordinates.csv')

# Load the CSV
soil_df = pd.read_csv(csv_path)

# Load API keys from environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini client after loading the key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")
    # You might want to raise an HTTPException or exit if the key is mandatory
    # raise HTTPException(status_code=500, detail="Gemini API key not configured.")

BASE_URL = "http://api.weatherapi.com/v1/forecast.json"
# ----------------------------- INPUT SCHEMAS -----------------------------
# Request schema for Crop Prediction API

# Define request schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    language: str = "English"   # default to English if not provided


class CropPlanInput(BaseModel):
    crop: str = Field(..., example="tomato")
    n: int = Field(..., ge=0)
    p: int = Field(..., ge=0)
    k: int = Field(..., ge=0)
    soil_ph: float = Field(..., ge=0.0, le=14.0)
    temperature: float
    humidity: float = Field(..., ge=0.0, le=100.0)
    rainfall: float = Field(..., ge=0.0)
    language: str = Field(..., example="marathi")

# Price Estimation Input Schema
class PriceEstimationRequest(BaseModel):
    district: int
    month: int
    market: int
    commodity: int
    variety: int
    agri_season: int
    climate_season: int

# Define a Pydantic model to accept the input data from the user
class YieldEstimationRequest(BaseModel):
    state: str
    district: str
    commodity: str
    season: str
    area_hectare: float  # Area in hectares

# Final response model
class SoilEstimate(BaseModel):
    n: float
    p: float
    k: float
    ph: float
    temperature: float
    humidity: float
    rainfall: float


 # Replace with your actual API key
class UserMessage(BaseModel):
    query: str

# Pydantic model for response structure (optional, for clarity)
class WeatherCondition(BaseModel):
    text: str
    icon: str

class DayForecast(BaseModel):
    date: str
    maxtemp_c: float
    mintemp_c: float
    avgtemp_c: float
    maxwind_kph: float
    totalprecip_mm: float
    avghumidity: float
    condition: WeatherCondition

class WeatherResponse(BaseModel):
    forecast: list[DayForecast]
    gemini_summary: str
# ----------------------------- STATIC MAPPINGS -----------------------------
# These lists will be used to map human-readable names to index values for model input

districts =['Ahmednagar', 'Akola', 'Amarawati', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Chattrapati Sambhajinagar', 'Dharashiv(Usmanabad)', 'Dhule', 'Gadchiroli', 'Hingoli', 'Jalana', 'Jalgaon', 'Kolhapur', 'Latur', 'Mumbai', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sholapur', 'Thane', 'Vashim', 'Wardha', 'Yavatmal']
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
markets =['ACF Agro Marketing', 'Aarni', 'Aatpadi', 'Achalpur', 'Aheri', 'Ahmednagar', 'Ahmedpur', 'Akhadabalapur', 'Akkalkot', 'Akkalkuwa', 'Akluj', 'Akola', 'Akole', 'Akot', 'Alibagh', 'Amalner', 'Amarawati', 'Ambad (Vadigodri)', 'Ambejaogai', 'Amrawati(Frui & Veg. Market)', 'Anajngaon', 'Armori(Desaiganj)', 'Arvi', 'Ashti', 'Ashti(Jalna)', 'Ashti(Karanja)', 'Aurad Shahajani', 'Ausa', 'BSK Krishi Bazar Private Ltd', 'Babhulgaon', 'Balapur', 'Baramati', 'Barshi', 'Barshi Takli', 'Barshi(Vairag)', 'Basmat', 'Basmat(Kurunda)', 'Beed', 'Bhadrawati', 'Bhagyoday Cotton and Agri Market', 'Bhandara', 'Bhivandi', 'Bhiwapur', 'Bhokar', 'Bhokardan', 'Bhokardan(Pimpalgaon Renu)', 'Bhusaval', 'Bodwad', 'Bori', 'Bori Arab', 'Buldhana', 'Buldhana(Dhad)', 'Chakur', 'Chalisgaon', 'Chandrapur', 'Chandrapur(Ganjwad)', 'Chandur Bazar', 'Chandur Railway', 'Chandvad', 'Chattrapati Sambhajinagar', 'Chikali', 'Chimur', 'Chopada', 'Cottoncity Agro Foods Private Ltd', 'Darwha', 'Daryapur', 'Deglur', 'Deoulgaon Raja', 'Deulgaon Raja Balaji Agro Marketing Private Market', 'Devala', 'Devani', 'Dhadgaon', 'Dhamngaon-Railway', 'Dharangaon', 'Dharashiv', 'Dharmabad', 'Dharni', 'Dhule', 'Digras', 'Dindori', 'Dindori(Vani)', 'Dondaicha', 'Dondaicha(Sindhkheda)', 'Dound', 'Dudhani', 'Fulmbri', 'Gadhinglaj', 'Gajanan Krushi Utpanna Bazar (India) Pvt Ltd', 'Gangakhed', 'Gangapur', 'Gevrai', 'Ghansawangi', 'Ghatanji', 'Ghoti', 'Gondpimpri', 'Gopal Krishna Agro', 'Hadgaon', 'Hadgaon(Tamsa)', 'Hari Har Khajagi Bazar Parisar', 'Higanghat Infrastructure Private Limited', 'Himalyatnagar', 'Hinganghat', 'Hingna', 'Hingoli', 'Hingoli(Kanegoan Naka)', 'Indapur', 'Indapur(Bhigwan)', 'Indapur(Nimgaon Ketki)', 'Islampur', 'J S K Agro Market', 'Jafrabad', 'Jagdamba Agrocare', 'Jai Gajanan Krishi Bazar', 'Jalana', 'Jalgaon', 'Jalgaon Jamod(Aasalgaon)', 'Jalgaon(Masawat)', 'Jalkot', 'Jalna(Badnapur)', 'Jamkhed', 'Jamner', 'Jamner(Neri)', 'Janata Agri Market (DLS Agro Infrastructure Pvt Lt', 'Jawala-Bajar', 'Jawali', 'Jaykissan Krushi Uttpan Khajgi Bazar', 'Jintur', 'Junnar', 'Junnar(Alephata)', 'Junnar(Narayangaon)', 'Junnar(Otur)', 'Kada', 'Kada(Ashti)', 'Kai Madhavrao Pawar Khajgi Krushi Utappan Bazar Sa', 'Kaij', 'Kalamb', 'Kalamb (Dharashiv)', 'Kalamnuri', 'Kalmeshwar', 'Kalvan', 'Kalyan', 'Kamthi', 'Kandhar', 'Kannad', 'Karad', 'Karanja', 'Karjat', 'Karjat(Raigad)', 'Karmala', 'Katol', 'Khamgaon', 'Khed', 'Khed(Chakan)', 'Khultabad', 'Kille Dharur', 'Kinwat', 'Kisan Market Yard', 'Kolhapur', 'Kolhapur(Malkapur)', 'Kopargaon', 'Koregaon', 'Korpana', 'Krushna Krishi Bazar', 'Kurdwadi', 'Kurdwadi(Modnimb)', 'Lakhandur', 'Lasalgaon', 'Lasalgaon(Niphad)', 'Lasalgaon(Vinchur)', 'Lasur Station', 'Late Vasantraoji Dandale Khajgi Krushi Bazar', 'Latur', 'Latur(Murud)', 'Laxmi Sopan Agriculture Produce Marketing Co Ltd', 'Loha', 'Lonand', 'Lonar', 'MS Kalpana Agri Commodities Marketing', 'Mahagaon', 'Maharaja Agresen Private Krushi Utappan Bazar Sama', 'Mahavir Agri Market', 'Mahavira Agricare', 'Mahesh Krushi Utpanna Bazar, Digras', 'Mahur', 'Majalgaon', 'Malegaon', 'Malegaon(Vashim)', 'Malharshree Farmers Producer Co Ltd', 'Malkapur', 'Manchar', 'Mandhal', 'Mangal Wedha', 'Mangaon', 'Mangrulpeer', 'Mankamneshwar Farmar Producer CoLtd Sanchalit Mank', 'Manmad', 'Manora', 'Mantha', 'Manwat', 'Marathawada Shetkari Khajgi Bazar Parisar', 'Maregoan', 'Mauda', 'Mehekar', 'Mohol', 'Morshi', 'Motala', 'Mudkhed', 'Mukhed', 'Mulshi', 'Mumbai', 'Mumbai- Fruit Market', 'Murbad', 'Murtizapur', 'Murud', 'Murum', 'N N Mundhada Agriculture Market Produce', 'Nagpur', 'Naigaon', 'Nampur', 'Nanded', 'Nandgaon', 'Nandgaon Khandeshwar', 'Nandura', 'Nandurbar', 'Narkhed', 'Nashik(Devlali)', 'Nasik', 'Navapur', 'Ner Parasopant', 'Newasa', 'Newasa(Ghodegaon)', 'Nilanga', 'Nira', 'Nira(Saswad)', 'Om Chaitanya Multistate Agro Purpose CoOp Society', 'Pachora', 'Pachora(Bhadgaon)', 'Paithan', 'Palam', 'Palghar', 'Palus', 'Pandhakawada', 'Pandharpur', 'Panvel', 'Parali Vaijyanath', 'Paranda', 'Parbhani', 'Parner', 'Parola', 'Parshiwani', 'Partur', 'Partur(Vatur)', 'Patan', 'Pathardi', 'Pathari', 'Patoda', 'Patur', 'Pavani', 'Pen', 'Perfect Krishi Market Yard Pvt Ltd', 'Phaltan', 'Pimpalgaon', 'Pimpalgaon Baswant(Saykheda)', 'Pombhurni', 'Pratap Nana Mahale Khajgi Bajar Samiti', 'Premium Krushi Utpanna Bazar', 'Pulgaon', 'Pune', 'Pune(Khadiki)', 'Pune(Manjri)', 'Pune(Moshi)', 'Pune(Pimpri)', 'Purna', 'Pusad', 'Rahata', 'Rahuri', 'Rahuri(Songaon)', 'Rahuri(Vambori)', 'Rajura', 'Ralegaon', 'Ramdev Krushi Bazaar', 'Ramtek', 'Rangrao Patil Krushi Utpanna Khajgi Bazar', 'Ratnagiri (Nachane)', 'Raver', 'Raver(Sauda)', 'Risod', 'Sakri', 'Samudrapur', 'Sangamner', 'Sangli', 'Sangli(Phale, Bhajipura Market)', 'Sangola', 'Sangrampur(Varvatbakal)', 'Sant Namdev Krushi Bazar,', 'Satana', 'Satara', 'Savner', 'Selu', 'Sengoan', 'Shahada', 'Shahapur', 'Shantilal Jain Agro', 'Shegaon', 'Shekari Krushi Khajgi Bazar', 'Shetkari Khajgi Bajar', 'Shetkari Khushi Bazar', 'Shevgaon', 'Shevgaon(Bodhegaon)', 'Shirpur', 'Shirur', 'Shivsiddha Govind Producer Company Limited Sanchal', 'Shree Rameshwar Krushi Market', 'Shree Sairaj Krushi Market', 'Shree Salasar Krushi Bazar', 'Shri Gajanan Maharaj Khajagi Krushi Utpanna Bazar', 'Shrigonda', 'Shrigonda(Gogargaon)', 'Shrirampur', 'Shrirampur(Belapur)', 'Sillod', 'Sillod(Bharadi)', 'Sindi', 'Sindi(Selu)', 'Sindkhed Raja', 'Sinner', 'Sironcha', 'Solapur', 'Sonpeth', 'Suragana', 'Tadkalas', 'Taloda', 'Tasgaon', 'Telhara', 'Tiwasa', 'Tuljapur', 'Tumsar', 'Udgir', 'Ulhasnagar', 'Umared', 'Umarga', 'Umari', 'Umarked(Danki)', 'Umarkhed', 'Umrane', 'Vadgaonpeth', 'Vaduj', 'Vadvani', 'Vai', 'Vaijpur', 'Vani', 'Varora', 'Varud', 'Varud(Rajura Bazar)', 'Vasai', 'Vashi New Mumbai', 'Vita', 'Vitthal Krushi Utpanna Bazar', 'Wardha', 'Washi (Dharashiv)', 'Washim', 'Washim(Ansing)', 'Yashika Agro Marketing', 'Yawal', 'Yeola', 'Yeotmal', 'ZariZamini']
commodities = ['Ajwan', 'Arecanut(Betelnut/Supari)', 'Arhar (Tur/Red Gram)(Whole)', 'Arhar Dal(Tur Dal)', 'Bajra(Pearl Millet/Cumbu)', 'Banana', 'Bengal Gram Dal (Chana Dal)', 'Bengal Gram(Gram)(Whole)', 'Bhindi(Ladies Finger)', 'Bitter gourd', 'Black Gram (Urd Beans)(Whole)', 'Black Gram Dal (Urd Dal)', 'Black pepper', 'Bottle gourd', 'Brinjal', 'Cabbage', 'Carrot', 'Cashewnuts', 'Castor Seed', 'Cauliflower', 'Chikoos(Sapota)', 'Chili Red', 'Chilly Capsicum', 'Coconut', 'Coriander(Leaves)', 'Corriander seed', 'Cotton', 'Cowpea (Lobia/Karamani)', 'Cucumbar(Kheera)', 'Cummin Seed(Jeera)', 'Drumstick', 'French Beans (Frasbean)', 'Garlic', 'Ginger(Dry)', 'Ginger(Green)', 'Grapes', 'Green Gram (Moong)(Whole)', 'Green Gram Dal (Moong Dal)', 'Green Peas', 'Guava', 'Jack Fruit', 'Jamun(Narale Hannu)', 'Jowar(Sorghum)', 'Kulthi(Horse Gram)', 'Lentil (Masur)(Whole)', 'Lime', 'Linseed', 'Maize', 'Mango', 'Methi(Leaves)', 'Mustard', 'Neem Seed', 'Niger Seed (Ramtil)', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Pomegranate', 'Potato', 'Pumpkin', 'Raddish', 'Ragi (Finger Millet)', 'Rice', 'Safflower', 'Sesamum(Sesame,Gingelly,Til)', 'Soanf', 'Soyabean', 'Spinach', 'Sugarcane', 'Sunflower', 'Tomato', 'Turmeric', 'Water Melon', 'Wheat']
varieties = ['1009 Kar', '147 Average', '1st Sort', '2nd Sort', 'Average (Whole)', 'Bansi', 'Black', 'Bold', 'DCH-32(Unginned)', 'Deshi Red', 'Deshi White', 'Desi', 'F.A.Q. Bold', 'Full Green', 'Gajjar', 'Green (Whole)', 'H-4(A) 27mm FIne', 'Hapus(Alphaso)', 'Hybrid', 'Jalgaon', 'Jowar ( White)', 'Jowar (Yellow)', 'Kabul Small', 'Kalyan', 'Kesari', 'Khandesh', 'LH-900', 'LRA', 'Local', 'Maharashtra 2189', 'Mogan Medium', 'N-44', 'Niger Seed', 'Other', 'Pole', 'RCH-2', 'Rajapuri', 'Red', 'Sharbati', 'Totapuri', 'Varalaxmi', 'White', 'White Fozi', 'Yellow', 'Yellow (Black)']
agri_seasons = ["Kharif", "Rabi", "Zaid"]
climate_seasons = ["Monsoon", "Post-Monsoon", "Summer", "Winter"]

#list for yeild data
statess = ['Maharashtra']  # Example states
Districtss = ['Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai suburban', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal', 'latur'] # Example districts
commoditiess =['Ajwain (Carom Seeds)', 'Aloe Vera', 'Arecanut (Betelnut)', 'Arhar/tur', 'Ashwagandha', 'Bajra', 'Bajra (Pearl Millet)', 'Banana', 'Barley', 'Ber (Indian Jujube)', 'Berseem', 'Bitter Gourd', 'Black Pepper', 'Bottle Gourd', 'Brinjal (Eggplant)', 'Cabbage', 'Carrot', 'Cashew Nut', 'Castor Seed', 'Castor seed', 'Cauliflower', 'Chana (Bengal Gram)', 'Chikoo (Sapota)', 'Chilli', 'Cluster Beans (Gavar)', 'Coconut', 'Coffee', 'Coriander', 'Coriander Seeds', 'Cotton', 'Cotton(lint)', 'Cucumber', 'Cumin (Jeera)', 'Custard Apple', 'Dill Seeds', 'Drumstick', 'Fennel (Saunf)', 'Fenugreek (Methi)', 'Fig (Anjeer)', 'French Beans', 'Garlic', 'Ginger', 'Gram', 'Grapes', 'Green Peas', 'Groundnut', 'Guava', 'Hybrid Napier Grass', 'Jackfruit', 'Jamun', 'Jowar', 'Jowar (Sorghum)', 'Kulthi (Horse Gram)', 'Lady Finger (Bhindi)', 'Lemon', 'Lemongrass', 'Linseed', 'Lobia (Cowpea)', 'Lucerne', 'Maize', 'Maize (For Fodder)', 'Mango', 'Masoor (Lentil)', 'Moong (Green Gram)', 'Moong(green gram)', 'Muskmelon', 'Mustard', 'Mustard Seeds', 'Neem', 'Niger (Ramtil)', 'Niger seed', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Pomegranate', 'Potato', 'Pumpkin', 'Radish', 'Ragi', 'Ragi (Finger Millet)', 'Rajma (Kidney Beans)', 'Rapeseed & Mustard', 'Rice', 'Safflower', 'Safflower (Kardi)', 'Sarpagandha', 'Sesame (Til)', 'Sesamum', 'Sorghum (For Fodder)', 'Soyabean', 'Soybean', 'Spinach', 'Sugarcane', 'Sunflower', 'Sweet Lime (Mosambi)', 'Tea', 'Tobacco', 'Tomato', 'Tulsi (Holy Basil)', 'Tur (Arhar/Red Gram)', 'Turmeric', 'Urad', 'Urad (Black Gram)', 'Watermelon', 'Wheat'] # Example commodities
seasonss =['Kharif', 'Rabi', 'Summer', 'Whole Year']  # Example seasons
# Language mapping for better language instruction
LANGUAGE_MAP = {
    'en': 'English',
    'hindi': 'Hindi (हिंदी)',
    'marathi': 'Marathi (मराठी)'
}






# Function to interact with Gemini API using genai client
def formatResponse(responseText: str) -> str:
    # Replace ***text*** with <h3>text</h3> (for headings)
    formattedResponse = re.sub(r'\*\*\*(.*?)\*\*\*', r'<h3>\1</h3>', responseText)
    
    # Replace **text** with <strong>text</strong> (for bold text)
    formattedResponse = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formattedResponse)
    
    # Replace *text* with <em>text</em> (for italic text)
    formattedResponse = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formattedResponse)
    
    # Replace section headers (e.g., "Crops:") with <h4>Crops:</h4>
    formattedResponse = re.sub(r'(^|\n)\s*([A-Za-z\s]+):', r'<h4>\2:</h4>', formattedResponse)
    
    # Replace bullet points (- item) with <li>item</li>
    formattedResponse = re.sub(r'\n- (.*?)(?=\n|$)', r'<li>\1</li>', formattedResponse)
    
    # Replace newlines with <br> for line breaks
    formattedResponse = re.sub(r'\n', r'<br>', formattedResponse)
    
    # Wrap bullet points in a <ul> tag if any <li> tags are present
    if '<li>' in formattedResponse:
        formattedResponse = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', formattedResponse)
    
    return formattedResponse


def extract_language_from_query(query: str) -> tuple[str, str]:
    """
    Extract language parameter from query string and return clean query + language
    """
    # Look for [lang=language] pattern
    lang_pattern = r'\[lang=([^\]]+)\]'
    match = re.search(lang_pattern, query)
    
    if match:
        lang_code = match.group(1).strip()
        # Remove the language parameter from the query
        clean_query = re.sub(lang_pattern, '', query).strip()
        return clean_query, lang_code
    
    return query, 'en'  # Default to English

def get_language_instruction(lang_code: str) -> str:
    """
    Get language-specific instruction for the AI model
    """
    if lang_code == 'en':
        return "Respond in English."
    elif lang_code == 'hindi':
        return "Respond in Hindi language (हिंदी में जवाब दें)."
    elif lang_code == 'marathi':
        return "Respond in Marathi language (मराठी भाषेत उत्तर द्या)."
    else:
        return "Respond in English."



# Function to interact with the Gemini API using genai client
def get_gemini_response(user_query: str, language: str = 'en') -> str:
    try:
        # Get language-specific instruction
        language_instruction = get_language_instruction(language)
        
        # Construct the prompt with language instruction
        prompt = (
    "You are a knowledgeable and friendly **female agricultural assistant** dedicated to helping farmers and agricultural professionals. "
    "Provide clear, concise, and practical advice in bullet points. "
    "Focus on actionable tips, best practices, and relevant information tailored to the user's question. "
    "Use simple language that is easy to understand and avoid unnecessary technical jargon. "
    "Speak with empathy and encouragement. "
    "If the question is vague or unclear, politely ask for clarification or suggest common topics the user might be interested in.\n\n"
    f"{language_instruction}\n\n"
    f"User question: {user_query}"
)
        
        # Initialize the model
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Generate the content
        response = model.generate_content(prompt)
        
        # Check and format the response
        if response.text:
            return formatResponse(response.text.strip())
        
        return "Sorry, I could not find an answer."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Gemini API: {str(e)}")

# Gemini multimodal query
def query_gemini_with_image(prompt: str, image_bytes: bytes) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        response = model.generate_content([
            prompt,
            image
        ])

        return response.text.strip()
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}"

def get_recent_weather_for_crop_model(lat: float, lon: float, elevation: float = 300, months: int = 5):
    location = Point(lat, lon, elevation)
    start = datetime(datetime.now().year - 2, 1, 1)
    end = datetime.now()

    monthly_data = Monthly(location, start, end).fetch()
    hourly_data = Hourly(location, start, end).fetch()

    # Handle humidity
    if 'rhum' in hourly_data.columns and not hourly_data['rhum'].isnull().all():
        hourly_data['month'] = hourly_data.index.to_period('M')
        monthly_humidity = hourly_data.groupby('month')['rhum'].mean()
    else:
        monthly_humidity = pd.Series(dtype=float)

    monthly_data.index = monthly_data.index.to_period('M')
    combined = monthly_data.join(monthly_humidity)

    combined = combined.rename(columns={
        'tavg': 'Avg_Temperature_C',
        'prcp': 'Total_Precipitation_mm',
        'rhum': 'Avg_Humidity_Percent'
    })

    recent_data = combined.iloc[-months:]
    
    avg_temp = round(recent_data['Avg_Temperature_C'].mean(skipna=True), 2) if 'Avg_Temperature_C' in recent_data else None
    avg_rainfall = round(recent_data['Total_Precipitation_mm'].mean(skipna=True), 2) if 'Total_Precipitation_mm' in recent_data else None
    avg_humidity = round(recent_data['Avg_Humidity_Percent'].mean(skipna=True), 2) if 'Avg_Humidity_Percent' in recent_data else None

    return {
        "temperature": float(avg_temp) if pd.notna(avg_temp) else None,
        "rainfall": float(avg_rainfall) if pd.notna(avg_rainfall) else None,
        "humidity": float(avg_humidity) if pd.notna(avg_humidity) else None
    }

async def fetch_weather_forecast(lat: float, lon: float) -> list:
    params = {
        "key": WEATHER_API_KEY,
        "q": f"{lat},{lon}",
        "days": 7,
        "aqi": "no",
        "alerts": "no"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(BASE_URL, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Weather API request failed")

        data = resp.json()

    # Extract useful forecast fields for farmers
    forecast_days = []
    for day in data.get("forecast", {}).get("forecastday", []):
        day_data = day.get("day", {})
        condition = day_data.get("condition", {})
        forecast_days.append({
            "date": day.get("date"),
            "maxtemp_c": day_data.get("maxtemp_c"),
            "mintemp_c": day_data.get("mintemp_c"),
            "avgtemp_c": day_data.get("avgtemp_c"),
            "maxwind_kph": day_data.get("maxwind_kph"),
            "totalprecip_mm": day_data.get("totalprecip_mm"),
            "avghumidity": day_data.get("avghumidity"),
            "condition": {
                "text": condition.get("text"),
                "icon": condition.get("icon")
            }
        })

    return forecast_days

def analyze_forecast_with_gemini(forecast_data: List[dict], language: str = "English") -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        prompt = (
            f"""
            You are an agriculture weather assistant. Analyze the following 7-day forecast and provide a summary for farmers.

            Focus on:
            - Temperature patterns
            - Rainfall and irrigation advice
            - Wind warnings or storm alerts
            - General agricultural suggestions

            Respond in {language}.

            Forecast:
            {forecast_data}
            """
        )
        response = model.generate_content(prompt)

        # Pass only the text content to formatResponse
        formatted_text = formatResponse(response.text)

        return formatted_text.strip()

    except Exception as e:
        return f"Gemini analysis failed: {str(e)}"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c
@app.get("/metrics")
def get_metrics():
    return {
        "message": "Basic API metrics are logged in 'api_metrics.log'",
        "example": "Check file for latency per endpoint."
    }
# Route for Home
@app.get("/predict")
def home():
    return {"message": "Welcome to Agrosarthi API"}

# Crop Prediction Route
@app.post("/predict/")
async def predict_crop(data: CropInput):
    language = data.language.strip().capitalize()

    # Prepare input for model prediction
    input_array = np.array([[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
    probs = crop_model.predict(input_array)[0]
    
    # Get top 3 crop indices
    top_indices = np.argsort(probs)[-3:][::-1]
    top_crops_english = [crop_label_mapping.get(i, "Unknown") for i in top_indices]

    # Initialize Gemini model
    gemini = genai.GenerativeModel("gemini-1.5-pro-latest")

    translated_crops = []
    explanations = []

    for crop in top_crops_english:
        # Translation
        translation_prompt = (
            f"Translate ONLY the crop name '{crop}' into {language}. "
            "Respond with ONLY the translated crop name, no explanations or extra text."
        )
        translation_response = gemini.generate_content(translation_prompt)
        translated_crop = translation_response.text.strip()
        translated_crops.append(translated_crop)

        # Individual explanation prompt for each crop
        explanation_prompt = (
            f"Given the following environmental conditions:\n"
            f"N={data.N}, P={data.P}, K={data.K}, Temperature={data.temperature}°C, "
            f"Humidity={data.humidity}%, pH={data.ph}, Rainfall={data.rainfall}mm.\n\n"
            f"Explain why the crop '{translated_crop}' is suitable for these conditions.\n"
            f"Respond in {language}."
        )
        explanation_response = gemini.generate_content(explanation_prompt)
        explanations.append(formatResponse(explanation_response.text.strip()))

    return {
        "top_crops": translated_crops,
        "explanations": explanations,
        "language": language
    }


@app.middleware("http")
async def log_api_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000  # in milliseconds

    logging.info(f"{request.method} {request.url.path} took {duration:.2f} ms with status {response.status_code}")
    return response

# Price Estimation Route
@app.post("/predict-price/")
async def estimate_price(request: PriceEstimationRequest):
    if price_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Prepare the input data for prediction
    input_features = np.array([[request.district, request.month - 1, request.market, request.commodity,
        request.variety, request.agri_season, request.climate_season]])

    try:
        # Predict price using the model
        predicted_price = price_model.predict(input_features)[0]

        result = {
            "district": districts[request.district],
            "month": months[request.month - 1],
            "market": markets[request.market],
            "commodity": commodities[request.commodity],
            "variety": varieties[request.variety],
            "agri_season": agri_seasons[request.agri_season],
            "climate_season": climate_seasons[request.climate_season],
            "predicted_price": predicted_price
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

@app.post("/predict-yield/")
async def estimate_yield(request: YieldEstimationRequest):
    # Check if model is loaded
    if yield_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Validate inputs
    if request.state not in statess:
        raise HTTPException(status_code=400, detail="Invalid state")
    if request.district not in Districtss:
        raise HTTPException(status_code=400, detail="Invalid district")
    if request.commodity not in commoditiess:
        raise HTTPException(status_code=400, detail="Invalid commodity")
    if request.season not in seasonss:
        raise HTTPException(status_code=400, detail="Invalid season")
    
    # Prepare the input features for prediction
    input_features = np.array([[
        statess.index(request.state), 
        Districtss.index(request.district), 
        commoditiess.index(request.commodity), 
        seasonss.index(request.season), 
        request.area_hectare
    ]])

    try:
        # Predict yield using the model (In ton/ha) - Replace this with your actual model prediction logic
        predicted_yield = np.random.random()  # Simulated prediction, replace with actual model prediction

        # Construct the result to return to the frontend
        result = {
            "state": request.state,
            "district": request.district,
            "commodity": request.commodity,
            "season": request.season,
            "area_hectare": request.area_hectare,
            "predicted_yield_ton_ha": predicted_yield  # Returning predicted yield in ton/ha
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Handle any unexpected errors
        return JSONResponse(content={"error": str(e)}, status_code=400)
  
@app.post("/query/")
async def query_chatbot(user_message: UserMessage):
    # Extract language and clean query
    clean_query, language = extract_language_from_query(user_message.query)
    
    # Get response from Gemini API with language support
    response = get_gemini_response(clean_query, language)
    
    return {
        "response": formatResponse(response),
        "language": language,
        "original_query": user_message.query,
        "processed_query": clean_query
    }

@app.post("/predict-disease/")
async def predict(
    crop_name: str = Form(...),
    language: str = Form(default="English"),
    file: UploadFile = File(...)
):
    crop_name = crop_name.strip()
    language = language.strip().capitalize()

    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    file_bytes = await file.read()

    # Prepare Gemini prompt with language support
    prompt = (
        f"""
        You are given an image of a {crop_name} plant showing visible symptoms. 
        Your task is to:

        1. Accurately identify the **disease or abnormality** affecting the plant.
        2. Provide the **exact name** of the disease.
        3. Suggest **clear, concise, and practical remedies or treatments** in bullet points.
        4. Keep the advice **short, actionable, and easy for a farmer to follow**.
        5. Respond in **{language}** only.

        Be accurate, avoid guessing, and only respond if symptoms are clearly identifiable.
        """
    )

    gemini_response = query_gemini_with_image(prompt, file_bytes)

    return JSONResponse({
        "crop": crop_name,
        "language": language,
        "predicted_disease_and_remedies":formatResponse(gemini_response)
    })
# API route
@app.get("/soil-estimate", response_model=SoilEstimate)
def get_soil_estimate(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    elevation: float = Query(300, description="Elevation in meters (default: 300)")
):
    # Find the nearest soil record based on lat/lon
    min_distance = float("inf")
    closest_row = None

    for _, row in soil_df.iterrows():
        dist = haversine(lat, lon, row['Latitude'], row['Longitude'])
        if dist < min_distance:
            min_distance = dist
            closest_row = row

    if closest_row is None:
        raise HTTPException(status_code=404, detail="No nearby soil data found.")

    weather = get_recent_weather_for_crop_model(lat, lon, elevation)
    if None in [weather['temperature'], weather['humidity'], weather['rainfall']]:
        raise HTTPException(status_code=500, detail="Incomplete weather data")

    return SoilEstimate(
        n=closest_row['Estimated_N'],
        p=closest_row['Estimated_P'],
        k=closest_row['Estimated_K'],
        ph=closest_row['Estimated_pH'],
        temperature=weather['temperature'],
        humidity=weather['humidity'],
        rainfall=weather['rainfall']
    )
@app.get("/weather-forecast")
async def weather_forecast(
    lat: float = Query(..., description="Latitude of the location"),
    lon: float = Query(..., description="Longitude of the location"),
    language: str = Query("English", description="Language for summary output")
):
    location_query = f"{lat},{lon}"

    params = {
        "key": WEATHER_API_KEY,
        "q": location_query,
        "days": 7,
        "aqi": "no",
        "alerts": "no"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch weather data")
        data = response.json()

    forecast_days = []
    for day in data.get("forecast", {}).get("forecastday", []):
        day_data = day.get("day", {})
        condition = day_data.get("condition", {})
        forecast_days.append({
            "date": day.get("date"),
            "maxtemp_c": day_data.get("maxtemp_c"),
            "mintemp_c": day_data.get("mintemp_c"),
            "avgtemp_c": day_data.get("avgtemp_c"),
            "maxwind_kph": day_data.get("maxwind_kph"),
            "totalprecip_mm": day_data.get("totalprecip_mm"),
            "avghumidity": day_data.get("avghumidity"),
            "condition": {
                "text": condition.get("text"),
                "icon": condition.get("icon")
            }
        })

    analysis = analyze_forecast_with_gemini(forecast_days, language=language)

    return {
        "location": data.get("location", {}),
        "forecast": forecast_days,
        "summary": formatResponse(analysis)
    }
@app.post("/generate-crop-plan", response_model=Dict[str, Any])
async def generate_crop_plan(data: CropPlanInput):
    prompt = f"""
You are an expert agronomist. Your task is to generate a **highly detailed and comprehensive** crop plan in JSON format.
The plan should be in the {data.language} language.

Inputs:
Crop: {data.crop}
N: {data.n}, P: {data.p}, K: {data.k}
Soil pH: {data.soil_ph}
Temperature: {data.temperature}°C
Humidity: {data.humidity}%
Rainfall: {data.rainfall}mm

Output format:
{{
  "name": "<Crop name in {data.language}>",
  "icon": "images/crops/{data.crop.lower().replace(" ", "_")}.svg",
  "steps": [
    {{
      "title": "<Stage Title in {data.language}>",
      "icon": "fa-seedling",
      "description": "<**Detailed, multi-sentence description** of the stage in {data.language}, covering its importance and key aspects.>",
      "timeframe": "<Timeframe>",
      "tasks": [
        {{
          "title": "<Task title in {data.language}>",
          "description": "<**Extremely detailed, step-by-step instructions** for the task in {data.language}, including specific measurements, methods, and best practices.>",
          "importance": "high/medium/low",
          "reminder": true
        }}
      ]
    }}
  ]
}}
Only return valid JSON. No extra explanations or markdown.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)  # ✅ FIXED: Removed 'await'

        if hasattr(response, 'text') and response.text:
            raw_text = response.text.strip()

        # ✅ Strip markdown code block if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]  # remove ```json
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]  # remove ending ```

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON: {str(e)}. Raw: {raw_text[:200]}")
        else:
            raise HTTPException(status_code=500, detail="Empty response from Gemini.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

