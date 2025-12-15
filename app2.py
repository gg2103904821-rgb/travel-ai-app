import streamlit as st
import pandas as pd
import json
import requests
import folium
from streamlit_folium import st_folium
from pinecone import Pinecone
from openai import AzureOpenAI
from datetime import datetime, timedelta
import base64
import random
from PIL import Image, ImageOps, ImageDraw, ImageFont
from io import BytesIO
import urllib.parse 
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ============================
# 1. Configuration & Keys
# ============================
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
    RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
except FileNotFoundError:
    st.error("没有找到 API Keys，请在 Streamlit Cloud 后台配置 Secrets！")
    st.stop()

INDEX_NAME = "travel-world-openai"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
AZURE_API_VERSION = "2023-05-15"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"
HOST_ATTRACTIONS = "travel-advisor.p.rapidapi.com"
HOST_HOTELS = "travel-advisor.p.rapidapi.com" 
HOST_WEATHER = "weather-api99.p.rapidapi.com"

# ============================
# 2. Styling & Helpers
# ============================
st.set_page_config(page_title="Wanderlust AI", layout="wide", page_icon="✈️")

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return None

banner_base64 = get_base64_of_bin_file("banner.png")
if banner_base64:
    banner_img_tag = f'<img src="data:image/png;base64,{banner_base64}" class="banner-img">'
else:
    banner_img_tag = '<img src="https://images.unsplash.com/photo-1476514525535-07fb3b4ae5f1?q=80&w=2070" class="banner-img">'

st.markdown(f"""
<style>
    .stApp {{
        background-color: #8EC5FC;
        background-image: linear-gradient(62deg, #8EC5FC 0%, #E0C3FC 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .white-card, .city-card, .plan-card, .hotel-card {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    .white-card *, .city-card *, .plan-card *, .hotel-card * {{
        color: #2c3e50 !important;
    }}
    h1, h2, h3, h4 {{ color: #1a1a1a !important; font-weight: 700 !important; }}
    p, li, small {{ color: #4a5568 !important; }}
    
    .banner-img {{ width: 100%; height: 180px; object-fit: cover; border-radius: 16px; margin-bottom: 20px; }}
    .card-img {{ width: 100%; height: 160px; object-fit: cover; border-radius: 12px; margin-bottom: 12px; }}
    
    .budget-box {{
        background: linear-gradient(135deg, #2c3e50 0%, #000000 100%) !important;
        color: white !important;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }}
    .budget-box * {{ color: white !important; }}
    
    .saved-plan-item {{
        background: white; padding: 10px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #6a11cb;
    }}
    
    a {{
        text-decoration: none;
        color: #6a11cb !important;
        font-weight: bold;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    
    .hotel-card {{
        padding: 15px;
        border: 1px solid #eee;
        transition: box-shadow 0.3s;
    }}
    .hotel-card:hover {{
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }}
    
    /* 酒店主图样式 */
    .hotel-main-img {{
        border-radius: 12px;
        height: 250px;
        object-fit: cover;
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper: Budget Calculator ---
def estimate_flight_cost(origin, destination):
    if not origin or not destination: return 500
    if origin.lower() in ["hong kong", "hk", "china"] and destination.lower() in ["tokyo", "osaka", "bangkok", "singapore", "seoul"]:
        return 350
    return 900 

def get_coordinates(location_name):
    try:
        geolocator = Nominatim(user_agent="wanderlust_ai_app")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geocode(location_name)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return 22.3193, 114.1694

# ============================
# 3. API Tools
# ============================

def get_location_id(city_name):
    try:
        headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": HOST_ATTRACTIONS}
        url = f"https://{HOST_ATTRACTIONS}/locations/search"
        resp = requests.get(url, headers=headers, params={"query": city_name, "limit": "1", "currency": "USD"}).json()
        if "data" in resp and resp["data"]:
            return resp["data"][0]["result_object"]["location_id"]
    except:
        pass
    return None

def fetch_city_details_for_plan(city_name):
    """
    Obtain authentic attraction/restaurant data, extracting only genuine image URLs
    """
    try:
        loc_id = get_location_id(city_name)
        if not loc_id: return f"Could not find location ID for {city_name}"
        
        headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": HOST_ATTRACTIONS}
        
        url_a = f"https://{HOST_ATTRACTIONS}/attractions/list"
        resp_a = requests.get(url_a, headers=headers, params={"location_id": loc_id, "limit": "6", "currency": "USD"}).json()
        
        url_r = f"https://{HOST_ATTRACTIONS}/restaurants/list"
        resp_r = requests.get(url_r, headers=headers, params={"location_id": loc_id, "limit": "6", "currency": "USD"}).json()
        
        items = []
        
        def process_items(data_list, type_label):
            if "data" in data_list:
                for i_item, item in enumerate(data_list["data"]):
                    if "name" in item:
                        name = item['name']
                        address = item.get('address', 'Address unavailable')
                        rating = item.get('rating', 'N/A')
                        num_reviews = item.get('num_reviews', '0')
                        price_level = item.get('price_level', 'N/A')
                        open_now_text = item.get('open_now_text', 'Hours not listed')

                        real_image_url = item.get('photo', {}).get('images', {}).get('original', {}).get('url', "")
                        if not real_image_url:
                             real_image_url = item.get('photo', {}).get('images', {}).get('large', {}).get('url', "N/A")
                        
                        map_query = urllib.parse.quote(f"{name} {city_name}")
                        map_link = f"https://www.google.com/maps/search/?api=1&query={map_query}"
                        
                        items.append(f"""
                        TYPE: {type_label}
                        NAME: {name}
                        ADDRESS: {address}
                        RATING: {rating} ({num_reviews} reviews)
                        PRICE_LEVEL: {price_level}
                        OPENING_HOURS: {open_now_text}
                        MAP_LINK: {map_link}
                        IMAGE_URL: {real_image_url} 
                        """)

        process_items(resp_a, "ATTRACTION")
        process_items(resp_r, "RESTAURANT")
        
        return "\n---\n".join(items)
    except Exception as e:
        return f"Using fallback data. Error: {str(e)}"

def search_hotels_smart(city_name, check_in_date, style, max_nightly_budget):
    """
    Acquire Authentic Hotels + Mandatory Safety Net Logic (Ensuring Results Every Time)
    """
    real_hotels = []
    
    try:
        loc_id = get_location_id(city_name)
        if loc_id:
            headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": HOST_ATTRACTIONS}
            url = f"https://{HOST_ATTRACTIONS}/hotels/list"

            resp = requests.get(url, headers=headers, params={
                "location_id": loc_id, 
                "limit": "30", 
                "currency": "USD",
                "sort": "recommended" 
            }).json()
            
            if "data" in resp:
                for i_item, item in enumerate(resp["data"]):
                    if "name" not in item: continue
                    
                    price_str = item.get("price", "$200") 
                    try:
                        clean_price = ''.join([c for c in price_str if c.isdigit()])
                        price = int(clean_price) if clean_price else 200
                    except: price = 200

                    real_image_url = item.get('photo', {}).get('images', {}).get('original', {}).get('url', "")
                    if not real_image_url:
                         real_image_url = item.get('photo', {}).get('images', {}).get('large', {}).get('url', "")
                    
                    booking_query = urllib.parse.quote(f"{item['name']} {city_name}")
                    booking_url = f"https://www.booking.com/searchresults.html?ss={booking_query}"

                    tags = []
                    rating = item.get("rating", "N/A")
                    if rating != "N/A" and float(rating) >= 4.5: tags.append("Top Rated")
                    if price > 400: tags.append("Luxury")
                    elif price < 150: tags.append("Value")
                    
                    real_hotels.append({
                        "name": item['name'],
                        "price": price,
                        "score": rating,
                        "tags": tags[:3],
                        "image": real_image_url,
                        "booking_url": booking_url
                    })
    except Exception as e:
        print(f"Hotel API Error: {e}")

    # Screening Logic
    filtered = [h for h in real_hotels if h['price'] <= max_nightly_budget]
    
    # If no results are found after filtering, return the cheapest item from the actual data.
    if not filtered and real_hotels:
        filtered = sorted(real_hotels, key=lambda x: x['price'])[:4]
        
    # If the API fails to return any data, generate simulated data (only as a last resort; typically not triggered)
    if not filtered:
        fallback_names = [f"{city_name} Grand Hotel", f"The {city_name} View", f"{city_name} City Center", "Royal Stay"]
        for i, name in enumerate(fallback_names):
            seed = random.randint(100, 999)
            fallback_img = f"https://loremflickr.com/600/400/hotel,luxury?random={seed}"
            booking_query = urllib.parse.quote(f"{name} {city_name}")
            
            filtered.append({
                "name": name,
                "price": 150 + (i * 50),
                "score": "8.5",
                "tags": ["Popular", "Fallback Data"],
                "image": fallback_img,
                "booking_url": f"https://www.booking.com/searchresults.html?ss={booking_query}"
            })

    # ranking
    if style == "Staycation": filtered.sort(key=lambda x: x['price'], reverse=True)
    elif style == "Budget": filtered.sort(key=lambda x: x['price'])
    else: 
        filtered.sort(key=lambda x: x['price'], reverse=True) 
            
    return filtered[:4]

# --- Helper: Pure Code-Generated Stamp Style (Retro) ---
def create_digital_stamp(image_file, title_text, location_text):
    try:
        img = Image.open(image_file).convert("RGBA")
    except:
        return None 

    target_w, target_h = 600, 800
    img = ImageOps.fit(img, (target_w, target_h), centering=(0.5, 0.5))
    
    paper_color = (250, 249, 245, 255) 
    border_width = 50 
    stamp_w = target_w + border_width * 2
    stamp_h = target_h + border_width * 2 + 120 
    
    stamp = Image.new("RGBA", (stamp_w, stamp_h), paper_color)
    stamp.paste(img, (border_width, border_width))
    
    draw = ImageDraw.Draw(stamp)
    draw.rectangle(
        [border_width-1, border_width-1, border_width+target_w, border_width+target_h], 
        outline="#D1D1D1", 
        width=1
    )
    
    mask = Image.new("L", (stamp_w, stamp_h), 255)
    draw_mask = ImageDraw.Draw(mask)
    r = 14
    for x in range(0, stamp_w, r*3):
        draw_mask.ellipse((x, -r, x+r*2, r), fill=0) 
        draw_mask.ellipse((x, stamp_h-r, x+r*2, stamp_h+r), fill=0) 
    for y in range(0, stamp_h, r*3):
        draw_mask.ellipse((-r, y, r, y+r*2), fill=0) 
        draw_mask.ellipse((stamp_w-r, y, stamp_w+r, y+r*2), fill=0) 
        
    stamp.putalpha(mask)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 46)
        font_loc = ImageFont.truetype("arial.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_loc = ImageFont.load_default()

    text_color = "#2C3E50"
    meta_color = "#7F8C8D"
    text_center_y = border_width + target_h + 50
    draw.text((stamp_w/2, text_center_y), title_text, fill=text_color, anchor="mm", font=font_title)
    
    line_y = text_center_y + 30
    draw.line([(stamp_w/2 - 30, line_y), (stamp_w/2 + 30, line_y)], fill="#BDC3C7", width=2)

    date_str = datetime.now().strftime("%Y.%m.%d")
    meta_text = f"{location_text.upper()} • {date_str}"
    draw.text((stamp_w/2, line_y + 30), meta_text, fill=meta_color, anchor="mm", font=font_loc)
    
    return stamp

# ============================
# 4. Agent Logic
# ============================
class TravelAgent:
    def __init__(self):
        self.client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_version=AZURE_API_VERSION, api_key=AZURE_API_KEY)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        self.mbti_map = {
            "INTJ": "quiet architecture logic", "INTP": "unique hidden-gems",
            "ENTJ": "luxury efficient", "ENTP": "adventure novelty",
            "INFJ": "spiritual quiet", "INFP": "artistic dreamy",
            "ENFJ": "culture social", "ENFP": "vibrant street-food",
            "ISTJ": "history tradition", "ISFJ": "cozy scenic",
            "ESTJ": "popular landmarks", "ESFJ": "food markets",
            "ISTP": "adventure outdoor", "ISFP": "nature aesthetics",
            "ESTP": "thrill nightlife", "ESFP": "party beach",
            "General": "popular scenic"
        }

    def recommend_cities(self, feelings, style, mbti):
        mbti_keywords = self.mbti_map.get(mbti, "")
        query = f"{feelings} {feelings} {style} {mbti_keywords}"
        res = self.client.embeddings.create(input=query, model=EMBEDDING_MODEL)
        results = self.index.query(vector=res.data[0].embedding, top_k=3, include_metadata=True)
        return [m['metadata'] for m in results['matches']]
    
    def analyze_image_for_stamp(self, image_bytes):
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt = """
        You are a poetic travel curator. 
        1. Analyze this image.
        2. Create a very short title (max 6 characters, e.g. 'Sunset Peak', 'Victoria Night').
        3. Write a 1-sentence poetic description (max 30 words).
        Return JSON: {"title": "...", "description": "..."}
        """
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def generate_plan_options(self, city, criteria):
        prompt = f"""
        User wants a trip to {city}.
        Inputs: {criteria['days']} days, {criteria['pax']} pax, Style: {criteria['style']}.
        User Vibe: "{criteria['feelings']}". MBTI: {criteria['mbti']}.
        CONFLICT RULE: PRIORITIZE VIBE over MBTI.
        Task: Create 2 DISTINCT concepts (Plan A vs Plan B).
        Output JSON: {{"Plan A": {{"title": "...", "description": "...", "highlights": ["..."]}}, "Plan B": ...}}
        """
        resp = self.client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        return json.loads(resp.choices[0].message.content)

    def generate_detailed_itinerary(self, city, plan_concept, criteria):
        real_data = fetch_city_details_for_plan(city)
        
        prompt = f"""
        Create a detailed {criteria['days']}-day itinerary for {city}.
        Concept: {plan_concept}
        
        Raw Data Provided (Contains Names, Addresses, Map Links, and IMAGE_URL, RATING, PRICE, HOURS):
        {real_data}
        
        CRITICAL FORMATTING RULES:
        1. **LOCATIONS**: For every attraction/restaurant, display:
           - The Name and Address as a link: `📍 [Name](Map_Link)`
           - The Single Real Image: `![Name](IMAGE_URL)` (If IMAGE_URL is 'N/A', do not show image).
           - Details Line: `🏠 Address: ... | ⭐ Rating: ... | 💰 Price Level: ... | 🕒 Hours: ...`
           (Use the exact fields provided. If fields are 'N/A' or missing, Estimate them based on the location type, e.g., "Estimated Price: $$").
           
        2. **TONE**: Engaging and clear.
        """
        
        resp = self.client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

# ============================
# 5. Main App Flow
# ============================

# Init State
if "step" not in st.session_state: st.session_state.step = 1
if "user_profile" not in st.session_state: st.session_state.user_profile = {}
if "trip_data" not in st.session_state: st.session_state.trip_data = {}
if "saved_plans" not in st.session_state: st.session_state.saved_plans = [] 
if "agent" not in st.session_state: st.session_state.agent = TravelAgent()
if "selected_hotel" not in st.session_state: st.session_state.selected_hotel = None
if "stamp_collection" not in st.session_state: st.session_state.stamp_collection = []

# Header
st.markdown(banner_img_tag, unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:10px; background:rgba(255,255,255,0.8); border-radius:10px; margin-bottom:20px;">
    <h1 style="color:#2C3E50; margin:0;">Wanderlust AI</h1>
    <p style="color:#555; margin:0;">Your Soul-Matched Travel Companion</p>
</div>
""", unsafe_allow_html=True)

# Sidebar (Saved Plans & Stamps)
with st.sidebar:
    st.header("👤 Profile")
    if st.session_state.user_profile:
        st.write(f"**User:** {st.session_state.user_profile.get('nickname')}")
        st.write(f"**MBTI:** {st.session_state.user_profile.get('mbti')}")
    
    st.divider()
    
    st.header("📂 My Saved Plans")
    if not st.session_state.saved_plans:
        st.info("No saved plans yet.")
    else:
        for i, plan in enumerate(st.session_state.saved_plans):
            with st.expander(f"📍 {plan['city']} ({plan['date']})"):
                st.markdown(plan['content'], unsafe_allow_html=True) 
                st.caption(f"Hotel: {plan.get('hotel', 'Not selected')}")
                st.caption(f"Total Budget: ${plan.get('total', 0):,.0f}")
                
    # === Sidebar Logic: Stamp Generation ===
    st.divider()
    st.header("📸 Memory Stamps")
    
    uploaded_file = st.file_uploader("Upload photo", type=['jpg', 'png', 'jpeg'], key="stamp_uploader")
    
    user_location = st.text_input("Location", "Hong Kong", key="stamp_loc")
    
    if uploaded_file and st.button("✨ Mint Stamp"):
        with st.spinner("Analyzing & Minting..."):
            # AI analysis
            bytes_data = uploaded_file.getvalue()
            ai_meta = st.session_state.agent.analyze_image_for_stamp(bytes_data)
            
            # Generate stamp images
            stamp_img = create_digital_stamp(
                uploaded_file, 
                ai_meta['title'], 
                user_location
            )
            
            # Retrieve latitude and longitude
            lat, lon = get_coordinates(user_location)
            
            # Place in the stamp album
            new_stamp_record = {
                "image": stamp_img,
                "title": ai_meta['title'],
                "desc": ai_meta['description'],
                "location": user_location,
                "lat": lat,
                "lon": lon,
                "time": datetime.now().strftime("%H:%M")
            }
            st.session_state.stamp_collection.append(new_stamp_record)
            
            st.success("Stamp added to your Journey Map!")

    # Preview the latest image + Download button
    if st.session_state.stamp_collection:
        latest = st.session_state.stamp_collection[-1]
        st.image(latest['image'], caption=f"Latest: {latest['title']}", use_container_width=True)
        
        buf = BytesIO()
        latest['image'].save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Stamp",
            data=byte_im,
            file_name=f"stamp_{latest['title']}.png",
            mime="image/png"
        )

progress = (st.session_state.step / 6) * 100
st.progress(int(progress))

# --- STEP 1: PERSONAL INFO ---
if st.session_state.step == 1:
    st.markdown("### 👤 Step 1: About You")
    with st.container():
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        nickname = col1.text_input("Nickname", value=st.session_state.user_profile.get("nickname", ""))
        gender = col2.selectbox("Gender", ["Female", "Male", "Other"])
        
        col3, col4 = st.columns(2)
        age = col3.number_input("Age", 18, 99, 25)
        mbti_options = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP", "General / Not Sure"]
        mbti = col4.selectbox("MBTI (Optional)", mbti_options, index=16)
        
        if st.button("Next ➡️"):
            if nickname:
                st.session_state.user_profile = {"nickname": nickname, "gender": gender, "age": age, "mbti": mbti.split(" ")[0]}
                st.session_state.step = 2
                st.rerun()
            else: st.error("Please enter a nickname.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: TRIP CRITERIA ---
elif st.session_state.step == 2:
    st.markdown("### ✈️ Step 2: Trip Criteria")
    with st.container():
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        origin = col1.text_input("Origin City", "Hong Kong")
        feelings = col2.text_input("Vibe / Feeling", placeholder="e.g. Quiet like 'Lost in Translation'")
        
        c1, c2, c3 = st.columns(3)
        daily_budget = c1.number_input("Daily Budget (USD/Person)", min_value=50, max_value=5000, value=250, step=50)
        
        days = c2.slider("Duration (Days)", 2, 10, 4)
        pax = c3.number_input("Travelers", 1, 10, 2)
        
        st.markdown("#### Travel Style")
        style = st.radio("Focus:", ["Citywalk", "Shopping", "Foodie", "Staycation", "Culture"], horizontal=True)
        
        if st.button("Find Matching Cities ✨"):
            if feelings:
                st.session_state.trip_data = {
                    "origin": origin, "feelings": feelings, 
                    "daily_budget": daily_budget, 
                    "days": days, "pax": pax, "style": style
                }
                with st.spinner("Analyzing world map..."):
                    cities = st.session_state.agent.recommend_cities(feelings, style, st.session_state.user_profile['mbti'])
                    st.session_state.recommended_cities = cities
                st.session_state.step = 3
                st.rerun()
            else: st.error("Please enter a vibe.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: CITY SELECTION ---
elif st.session_state.step == 3:
    st.markdown("### 🏙️ Step 3: Top Matches")
    cols = st.columns(3)
    for i, city_data in enumerate(st.session_state.recommended_cities):
        with cols[i]:
            c_name = city_data.get('city')
            img_url = f"https://loremflickr.com/400/300/{c_name},travel"
            
            st.markdown(f"""
            <div class="city-card">
                <img src="{img_url}" class="card-img">
                <h3>{c_name}</h3>
                <p>{city_data.get('country')}</p>
                <p style="font-size:0.8rem;">{city_data.get('description')[:50]}...</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Choose {c_name}", key=f"c_{i}", use_container_width=True):
                st.session_state.selected_city = c_name
                with st.spinner("Drafting concepts..."):
                    criteria = {**st.session_state.user_profile, **st.session_state.trip_data}
                    plans = st.session_state.agent.generate_plan_options(c_name, criteria)
                    st.session_state.plan_concepts = plans
                st.session_state.step = 4
                st.rerun()
    if st.button("Back"): st.session_state.step = 2; st.rerun()

# --- STEP 4: PLAN SELECTION ---
elif st.session_state.step == 4:
    city = st.session_state.selected_city
    st.markdown(f"### 🗺️ Step 4: Choose Style for {city}")
    
    concepts = st.session_state.plan_concepts
    for plan_name, details in concepts.items():
        with st.container():
            st.markdown('<div class="plan-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(f"{plan_name}: {details['title']}")
                st.write(details['description'])
                st.caption(f"Highlights: {', '.join(details['highlights'])}")
            with c2:
                st.write("")
                if st.button(f"Select {plan_name}", key=plan_name):
                    st.session_state.selected_plan_name = plan_name
                    with st.spinner(f"Contacting Travel Advisor API for {city}..."):
                        criteria = {**st.session_state.user_profile, **st.session_state.trip_data}
                        detail = st.session_state.agent.generate_detailed_itinerary(city, details['title'], criteria)
                        st.session_state.final_itinerary = detail
                    st.session_state.step = 5
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 5: ITINERARY + HOTELS ---
elif st.session_state.step == 5:
    city = st.session_state.selected_city
    data = st.session_state.trip_data
    
    if "current_hotel_list" not in st.session_state or st.session_state.current_hotel_list is None:
        hotel_budget_max = data['daily_budget'] * 0.7 
        st.session_state.current_hotel_list = search_hotels_smart(
            city, datetime.now().strftime("%Y-%m-%d"), data['style'], hotel_budget_max
        )

    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.markdown(f"## 🗓️ Itinerary: {st.session_state.selected_plan_name}")
        
        if st.session_state.selected_hotel:
            sh = st.session_state.selected_hotel
            st.info(f"""
            ✅ **Selected Hotel:** {sh['name']}
            
            💰 Price: **${sh['price']}**/night  |  ⭐ Rating: **{sh['score']}**
            """)
        else:
            st.warning("🛏️ You haven't selected a hotel yet. Pick one from the right side.")

        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.final_itinerary, unsafe_allow_html=True) # HTML Allowed here
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("💾 Save Plan to Sidebar"):
            plan_record = {
                "city": city,
                "date": datetime.now().strftime("%m-%d %H:%M"),
                "content": st.session_state.final_itinerary,
                "hotel": st.session_state.selected_hotel['name'] if st.session_state.selected_hotel else "None",
                "total": st.session_state.total_cost if 'total_cost' in st.session_state else 0
            }
            st.session_state.saved_plans.append(plan_record)
            st.toast("Plan saved! Check the Sidebar.")
            
            import time
            time.sleep(0.5)
            st.rerun()

    with col_r:
        st.markdown("### 🏨 Recommended Hotels")
        
        hotels = st.session_state.current_hotel_list

        if not hotels: st.warning("We couldn't find hotels via API. Please check your network.")
        
        for h in hotels:
            with st.container():
                st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
                
                st.markdown(f"#### 🏨 [{h['name']}]({h['booking_url']})")
                st.caption("Click name to book on Booking.com ↗")

                if h['image']:
                    st.markdown(f"""
                    <img src="{h['image']}" style="width:100%; height:200px; object-fit:cover; border-radius:12px; margin-bottom:10px;">
                    """, unsafe_allow_html=True)
                else:
                    st.caption("No image available")
                
                st.markdown(f"""
                <div class="hotel-info">
                    <p>⭐ <b>{h['score']}</b> • <b style="color:#e67e22; font-size:1.1em;">${h['price']}</b>/night</p>
                    <small>{', '.join(h['tags'])}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                is_sel = (st.session_state.selected_hotel is not None) and (st.session_state.selected_hotel['name'] == h['name'])
                
                if st.button("✅ Selected" if is_sel else f"Add (${h['price']})", key=f"btn_{h['name']}", disabled=is_sel):
                    st.session_state.selected_hotel = h
                    st.rerun()
        
        st.markdown("---")
        
        if st.session_state.selected_hotel:
            h_price = st.session_state.selected_hotel['price']
            nights = data['days'] - 1
            flight_est = estimate_flight_cost(data['origin'], city)
            hotel_total = h_price * nights
            daily_spend = (data['daily_budget'] - h_price) * data['days'] * data['pax']
            if daily_spend < 0: daily_spend = 50 * data['days'] * data['pax']
            
            total_est = flight_est + hotel_total + daily_spend
            st.session_state.total_cost = total_est
            
            st.markdown(f"""
            <div class="budget-box">
                <small>Flights (Est): ${flight_est}</small><br>
                <small>Hotels ({nights} nights): ${hotel_total}</small><br>
                <small>Food/Activities: ${daily_spend:.0f}</small><br>
                <hr>
                <h2>Total: ${total_est:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)


    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Start New Trip"):
            st.session_state.step = 1
            st.session_state.selected_hotel = None
            st.session_state.current_hotel_list = None
            st.rerun()
    with col_btn2:
        if st.button("🗺️ View Journey Map (Step 6) ➡️"):
            st.session_state.step = 6
            st.rerun()
            
# --- STEP 6: JOURNEY MAP ---
elif st.session_state.step == 6:
    st.markdown("## 🌍 My Journey Map & Album")
    st.markdown("Your digital footprint, immortalized as stamps.")
    
    col_map, col_album = st.columns([2, 1])
    
    with col_map:
        st.markdown("### 📍 Interaction Map")
        
        if not st.session_state.stamp_collection:
            st.info("No stamps yet! Upload photos in the Sidebar to start tracking your journey.")
            m = folium.Map(location=[22.3193, 114.1694], zoom_start=11)
        else:
            # Initialize the map, setting the center point to the location of the first stamp.
            start_loc = [st.session_state.stamp_collection[0]['lat'], st.session_state.stamp_collection[0]['lon']]
            m = folium.Map(location=start_loc, zoom_start=13)
            
            # Prepare a list of trajectory coordinate points
            route_coords = []
            
            # Traverse the stamp album to mark the points
            for idx, stamp in enumerate(st.session_state.stamp_collection):
                coord = [stamp['lat'], stamp['lon']]
                route_coords.append(coord)
                
                popup_html = f"""
                <b>{stamp['title']}</b><br>
                {stamp['location']}<br>
                <i>{stamp['desc']}</i>
                """
                
                folium.Marker(
                    location=coord,
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"{idx+1}. {stamp['title']}",
                    icon=folium.Icon(color="purple", icon="camera", prefix="fa")
                ).add_to(m)
            
            # Plot trajectory lines
            if len(route_coords) > 1:
                folium.PolyLine(
                    route_coords,
                    color="#6a11cb",
                    weight=4,
                    opacity=0.8,
                    dash_array='10'
                ).add_to(m)

        st_folium(m, width="100%", height=500)

    with col_album:
        st.markdown("### 📒 Stamp Album")
        
        if not st.session_state.stamp_collection:
            st.write("waiting for memories...")
        else:
            for stamp in reversed(st.session_state.stamp_collection):
                with st.container():
                    st.markdown(f"**{stamp['title']}** | *{stamp['location']}*")
                    st.image(stamp['image'], use_container_width=True)
                    st.caption(f"💭 {stamp['desc']}")
                    st.divider()
                    
    if st.button("⬅️ Back to Itinerary"):
        st.session_state.step = 5
        st.rerun()
