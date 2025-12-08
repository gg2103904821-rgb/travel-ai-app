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

# ============================
# 1. Configuration & Keys
# ============================
# 从 Streamlit Secrets 读取 Key
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
    RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
except FileNotFoundError:
    st.error("没有找到 API Keys，请在 Streamlit Cloud 后台配置 Secrets！")
    st.stop()

# 其他不需要保密的配置可以保留
INDEX_NAME = "travel-world-openai"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
AZURE_API_VERSION = "2023-05-15"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"
HOST_ATTRACTIONS = "travel-advisor.p.rapidapi.com"
HOST_HOTELS = "booking-com15.p.rapidapi.com"
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
    /* 全局背景 */
    .stApp {{
        background-color: #8EC5FC;
        background-image: linear-gradient(62deg, #8EC5FC 0%, #E0C3FC 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}

    /* 卡片通用样式 - 强制白底黑字 */
    .white-card, .city-card, .plan-card, .hotel-card {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }}
    
    /* 暴力修复字体颜色 */
    .white-card *, .city-card *, .plan-card *, .hotel-card * {{
        color: #2c3e50 !important;
    }}
    h1, h2, h3, h4 {{ color: #1a1a1a !important; font-weight: 700 !important; }}
    p, li, small {{ color: #4a5568 !important; }}

    /* 图片样式 */
    .banner-img {{ width: 100%; height: 180px; object-fit: cover; border-radius: 16px; margin-bottom: 20px; }}
    .card-img {{ width: 100%; height: 160px; object-fit: cover; border-radius: 12px; margin-bottom: 12px; }}

    /* 预算黑盒 */
    .budget-box {{
        background: linear-gradient(135deg, #2c3e50 0%, #000000 100%) !important;
        color: white !important;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }}
    .budget-box * {{ color: white !important; }}
    
    /* 保存的计划样式 */
    .saved-plan-item {{
        background: white; padding: 10px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #6a11cb;
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper: Budget Calculator ---
def estimate_flight_cost(origin, destination):
    if not origin or not destination: return 500
    if origin.lower() in ["hong kong", "hk", "china"] and destination.lower() in ["tokyo", "osaka", "bangkok", "singapore", "seoul"]:
        return 350
    return 900 

# ============================
# 3. API Tools (Fixed & Enhanced)
# ============================

def fetch_city_details_for_plan(city_name):
    """
    真正调用 Travel Advisor API 并提取图片 URL
    """
    try:
        headers = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": HOST_ATTRACTIONS}
        
        # 1. 获取 Location ID
        url_s = f"https://{HOST_ATTRACTIONS}/locations/search"
        resp = requests.get(url_s, headers=headers, params={"query": city_name, "limit": "1", "currency": "USD"}).json()
        loc_id = resp["data"][0]["result_object"]["location_id"]
        
        # 2. 获取景点 (Attractions)
        url_a = f"https://{HOST_ATTRACTIONS}/attractions/list"
        resp_a = requests.get(url_a, headers=headers, params={"location_id": loc_id, "limit": "4", "currency": "USD"}).json()
        
        # 3. 获取餐厅 (Restaurants)
        url_r = f"https://{HOST_ATTRACTIONS}/restaurants/list"
        resp_r = requests.get(url_r, headers=headers, params={"location_id": loc_id, "limit": "4", "currency": "USD"}).json()
        
        items = []
        
        # 处理景点
        if "data" in resp_a:
            for item in resp_a["data"]:
                if "name" in item:
                    # 提取真实图片
                    img = item.get('photo', {}).get('images', {}).get('large', {}).get('url', "")
                    if not img: 
                        img = f"https://loremflickr.com/400/300/{item['name'].split()[0]},landmark"
                    
                    items.append(f"ATTRACTION: {item['name']} | Rating: {item.get('rating')} | Image: {img}")

        # 处理餐厅
        if "data" in resp_r:
            for item in resp_r["data"]:
                if "name" in item:
                    img = item.get('photo', {}).get('images', {}).get('large', {}).get('url', "")
                    if not img: 
                        img = f"https://loremflickr.com/400/300/food,restaurant"
                    items.append(f"RESTAURANT: {item['name']} | Rating: {item.get('rating')} | Image: {img}")
        
        return "\n".join(items)
    except Exception as e:
        return f"Using fallback data. Error: {str(e)}"

def search_hotels_smart(city_name, check_in_date, style, max_nightly_budget):
    """
    智能酒店搜索 + 稳定图片生成
    """
    all_hotels = []
    prefixes = [f"{city_name} Grand", f"The {city_name} View", f"{city_name} Boutique", "City Center Inn", "Backpacker Hostel", "Luxury Palace", "Comfort Stay", "Urban Hub"]
    
    for name in prefixes:
        base = 50
        multiplier = random.uniform(1, 10) 
        price = int(base * multiplier)
        
        tags = ["WiFi"]
        if price > 250: tags += ["Pool", "Spa", "Luxury"]
        if price < 100: tags += ["Budget", "Value"]
        
        score = round(random.uniform(7.5, 9.8), 1)
        
        # 稳定随机图
        rand_id = random.randint(1, 1000)
        img_keyword = "luxury,hotel" if price > 200 else "hostel,room" if price < 80 else "hotel,room"
        img_url = f"https://loremflickr.com/400/300/{img_keyword}?random={rand_id}"
        
        all_hotels.append({
            "name": name,
            "price": price,
            "score": score,
            "tags": tags,
            "image": img_url
        })
    
    # 筛选逻辑
    filtered_hotels = [h for h in all_hotels if h['price'] <= max_nightly_budget]
    if not filtered_hotels: filtered_hotels = sorted(all_hotels, key=lambda x: x['price'])[:3]
    
    if style == "Staycation": filtered_hotels.sort(key=lambda x: x['price'], reverse=True)
    elif style == "Budget": filtered_hotels.sort(key=lambda x: x['price'])
    else: filtered_hotels.sort(key=lambda x: x['score'], reverse=True)
        
    return filtered_hotels[:4]

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
        
        Data Provided (Includes Image URLs):
        {real_data}
        
        CRITICAL RULES:
        1. **IMAGES**: You MUST display images for every attraction and restaurant mentioned.
           Use Markdown format: `![Name](ImageURL)`
           Use the exact URLs provided in the 'Data Provided' section.
        2. **PRICING**: Use specific numbers (e.g. "Ticket: $15").
        3. **HOURS**: Include opening hours.
        4. **FORMAT**: Use clean Markdown.
        """
        resp = self.client.chat.completions.create(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content

# ============================
# 5. Main App Flow
# ============================

# Init State (⚠️ 修复：在这里添加了 selected_hotel 的初始化)
if "step" not in st.session_state: st.session_state.step = 1
if "user_profile" not in st.session_state: st.session_state.user_profile = {}
if "trip_data" not in st.session_state: st.session_state.trip_data = {}
if "saved_plans" not in st.session_state: st.session_state.saved_plans = [] 
if "agent" not in st.session_state: st.session_state.agent = TravelAgent()
if "selected_hotel" not in st.session_state: st.session_state.selected_hotel = None # ✅ 修复了报错源

# Header
st.markdown(banner_img_tag, unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:10px; background:rgba(255,255,255,0.8); border-radius:10px; margin-bottom:20px;">
    <h1 style="color:#2C3E50; margin:0;">Wanderlust AI</h1>
    <p style="color:#555; margin:0;">Your Soul-Matched Travel Companion</p>
</div>
""", unsafe_allow_html=True)

# Sidebar (Saved Plans Here!)
with st.sidebar:
    st.header("👤 Profile")
    if st.session_state.user_profile:
        st.write(f"**User:** {st.session_state.user_profile.get('nickname')}")
        st.write(f"**MBTI:** {st.session_state.user_profile.get('mbti')}")
    
    st.divider()
    
    # ⚠️ 保存计划查看区
    st.header("📂 My Saved Plans")
    if not st.session_state.saved_plans:
        st.info("No saved plans yet.")
    else:
        for i, plan in enumerate(st.session_state.saved_plans):
            with st.expander(f"📍 {plan['city']} ({plan['date']})"):
                st.markdown(plan['content']) # 显示保存的行程
                st.caption(f"Hotel: {plan.get('hotel', 'Not selected')}")
                st.caption(f"Total Budget: ${plan.get('total', 0):,.0f}")

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
        budget_level = c1.select_slider("Budget Level", options=["Budget", "Standard", "Luxury"])
        days = c2.slider("Duration (Days)", 2, 10, 4)
        pax = c3.number_input("Travelers", 1, 10, 2)
        
        st.markdown("#### Travel Style")
        style = st.radio("Focus:", ["Citywalk", "Shopping", "Foodie", "Staycation", "Culture"], horizontal=True)
        
        budget_map = {"Budget": 150, "Standard": 350, "Luxury": 800}
        
        if st.button("Find Matching Cities ✨"):
            if feelings:
                st.session_state.trip_data = {
                    "origin": origin, "feelings": feelings, 
                    "budget_level": budget_level, "daily_budget": budget_map[budget_level],
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
    
    # === [关键修复 1] 初始化酒店列表缓存 ===
    # 只有当缓存里没有酒店列表时，才调用 API/随机生成函数
    # 这样保证每次刷新页面，右边的推荐列表是固定的，不会乱变
    if "current_hotel_list" not in st.session_state or st.session_state.current_hotel_list is None:
        hotel_budget_max = data['daily_budget'] * 0.5 
        st.session_state.current_hotel_list = search_hotels_smart(
            city, datetime.now().strftime("%Y-%m-%d"), data['style'], hotel_budget_max
        )

    col_l, col_r = st.columns([2, 1])
    
    # === 左侧：行程详情 + 已选酒店显示 ===
    with col_l:
        st.markdown(f"## 🗓️ Itinerary: {st.session_state.selected_plan_name}")
        
        # 显示已选酒店
        if st.session_state.selected_hotel:
            sh = st.session_state.selected_hotel
            # 使用 info 框高亮显示
            st.info(f"""
            ✅ **Selected Hotel:** {sh['name']}
            
            💰 Price: **${sh['price']}**/night  |  ⭐ Rating: **{sh['score']}**
            """)
        else:
            st.warning("🛏️ You haven't selected a hotel yet. Pick one from the right side.")

        # 行程卡片
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.final_itinerary)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 保存按钮
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
            
            # 强制刷新以更新侧边栏
            import time
            time.sleep(0.5)
            st.rerun()

    # === 右侧：酒店推荐 ===
    with col_r:
        st.markdown("### 🏨 Recommended Hotels")
        
        # [关键修复 2] 从 session_state 读取固定的列表，而不是重新生成
        hotels = st.session_state.current_hotel_list
        
        if not hotels: st.warning("Budget too low for hotels.")
        
        for h in hotels:
            with st.container():
                st.markdown('<div class="hotel-card">', unsafe_allow_html=True)
                st.image(h['image'], use_container_width=True)
                st.markdown(f"""
                <div class="hotel-info">
                    <h4>{h['name']}</h4>
                    <p>⭐ {h['score']} • <b>${h['price']}</b>/night</p>
                    <small>{', '.join(h['tags'])}</small>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 交互逻辑
                # 确保 is_sel 是布尔值
                is_sel = (st.session_state.selected_hotel is not None) and (st.session_state.selected_hotel['name'] == h['name'])
                
                # 按钮逻辑
                if st.button("✅ Selected" if is_sel else f"Add (${h['price']})", key=f"btn_{h['name']}", disabled=is_sel):
                    st.session_state.selected_hotel = h
                    st.rerun() # 立即刷新，触发左侧 info 更新
        
        st.markdown("---")
        
        # 预算计算显示区域
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

    # 重置按钮需要顺便清空酒店缓存
    if st.button("🔄 Start New Trip"):
        st.session_state.step = 1
        st.session_state.selected_hotel = None
        st.session_state.current_hotel_list = None # [关键修复 3] 清空缓存，下次生成新的
        st.rerun().rerun()