import os
import json
import requests
from typing import Dict, List, Any, Literal, Optional, Annotated
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import time
from langgraph.graph.message import AnyMessage, add_messages

# API Keys setup
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "travel3"

langsmit_api_key = "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"
geoapify_api = "48f7eaa23139406a903a9c7b351edc4b"
OPENWEATHER_API_KEY = "f09c8818b978dbb75c0a83da4c21767b"
TICKETMASTER_API_KEY = "mA48YKgpK1h9SvqPzxWJri8beg9Phv8B"

os.environ["GOOGLE_API_KEY"] = "AIzaSyDaM0twsGt6Rv5M2pY4ze4lZlKc7IQWuiQ"
os.environ["TAVILY_API_KEY"] = "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"

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@dataclass
class HotelInfo:
    name: str
    address: str
    rating: float
    price_per_night: float
    total_price: float
    image_url: str
    amenities: List[str]
    distance_from_center: float
    phone: str
    website: str

@dataclass
class TransportInfo:
    mode: str
    departure_time: str
    arrival_time: str
    duration: str
    price_per_person: float
    total_price: float
    operator: str
    route: str
    booking_url: str

@dataclass
class WeatherInfo:
    date: str
    temperature_high: float
    temperature_low: float
    condition: str
    humidity: int
    wind_speed: float
    precipitation_chance: int
    recommendation: str

@dataclass
class EventInfo:
    name: str
    date: str
    time: str
    venue: str
    price_range: str
    category: str
    description: str
    booking_url: str

@dataclass
class FoodRecommendation:
    restaurant_name: str
    cuisine_type: str
    location: str
    price_range: str
    rating: float
    specialties: List[str]
    meal_type: str
    estimated_cost: float
    description: str

@dataclass
class BudgetBreakdown:
    total_budget: float
    accommodation: float
    transport: float
    activities: float
    food: float
    emergency: float
    remaining: float
    daily_allowance: float

# Enhanced State with completion tracking
class ExtendedMessagesState(MessagesState):
    hotels: List[HotelInfo] = []
    transport_options: List[TransportInfo] = []
    weather_forecast: List[WeatherInfo] = []
    events: List[EventInfo] = []
    food_recommendations: List[FoodRecommendation] = []
    budget_breakdown: Optional[BudgetBreakdown] = None
    current_plan: Dict[str, Any] = {}
    supervisor_feedback: str = ""
    alerts: List[str] = []
    # NEW: Track completion status to prevent unnecessary iterations
    completed_tasks: Dict[str, bool] = {}
    iteration_count: int = 0
    max_iterations: int = 5  # Prevent infinite loops

def get_place_coordinates(city: str, api_key: str) -> tuple:
    """Helper function to get coordinates for a city"""
    try:
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {"text": city, "apiKey": api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                return coords[1], coords[0]  # lat, lon
    except:
        pass
    return 0, 0  # fallback

@tool
def recommend_activities_tool(city: str, category: str, radius: int = 5000, limit: int = 10) -> List[Dict[str, Any]]:
    """Recommends activities in a city using Geoapify Places API"""
    api_key = "48f7eaa23139406a903a9c7b351edc4b"
    lat, lon = get_place_coordinates(city, api_key)

    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": category,
        "filter": f"circle:{lon},{lat},{radius}",
        "bias": f"proximity:{lon},{lat}",
        "limit": limit,
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return [{"message": f"No {category} found near {city}"}]

        results = response.json()
        features = results.get("features", [])

        if not features:
            return [{"message": f"No {category} found near {city}"}]

        activities = []
        for feature in features:
            prop = feature.get("properties", {})
            activities.append({
                "name": prop.get("name", "No name"),
                "category": prop.get("categories", []),
                "description": prop.get("datasource", {}).get("raw", {}).get("description", "No description"),
                "address": prop.get("formatted", "No address"),
                "website": prop.get("website", ""),
                "open_hours": prop.get("opening_hours", "")
            })

        return activities
    except Exception as e:
        return [{"message": f"Error finding {category} in {city}"}]

@tool
def search_events(destination: str, start_date: str, end_date: str) -> List[Dict]:
    """Search for local events using Ticketmaster API"""
    try:
        events_url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {
            "apikey": TICKETMASTER_API_KEY,
            "city": destination,
            "startDateTime": f"{start_date}T00:00:00Z",
            "endDateTime": f"{end_date}T23:59:59Z",
            "size": 10,
            "sort": "date,asc"
        }

        response = requests.get(events_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            events = []

            if "_embedded" in data and "events" in data["_embedded"]:
                for event in data["_embedded"]["events"][:5]:
                    price_range = "Price varies"
                    if "priceRanges" in event:
                        min_price = event["priceRanges"][0].get("min", 0)
                        max_price = event["priceRanges"][0].get("max", 0)
                        price_range = f"${min_price}-${max_price}"

                    venue = "Venue TBA"
                    if "_embedded" in event and "venues" in event["_embedded"]:
                        venue = event["_embedded"]["venues"][0]["name"]

                    events.append({
                        "name": event["name"],
                        "date": event["dates"]["start"]["localDate"],
                        "time": event["dates"]["start"].get("localTime", "Time TBA"),
                        "venue": venue,
                        "price_range": price_range,
                        "category": event["classifications"][0]["segment"]["name"] if "classifications" in event else "Entertainment",
                        "description": event.get("info", "No description available"),
                        "booking_url": event.get("url", "")
                    })
            return events
        return []
    except Exception as e:
        return []

@tool
def get_weather_forecast(destination: str, start_date: str, days: int) -> List[Dict]:
    """Get weather forecast using OpenWeatherMap free API"""
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={destination}&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url, timeout=10)
        
        if geo_response.status_code == 200:
            geo_data = geo_response.json()
            if geo_data:
                lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
                
                weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
                weather_response = requests.get(weather_url, timeout=10)
                
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    
                    daily_forecasts = []
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    
                    for i in range(min(days, 5)):
                        forecast_date = start_dt + timedelta(days=i)
                        date_str = forecast_date.strftime("%Y-%m-%d")
                        
                        day_forecasts = [
                            f for f in weather_data['list'] 
                            if f['dt_txt'].startswith(date_str)
                        ]
                        
                        if day_forecasts:
                            temps = [f['main']['temp'] for f in day_forecasts]
                            conditions = [f['weather'][0]['description'] for f in day_forecasts]
                            humidity = day_forecasts[0]['main']['humidity']
                            wind_speed = day_forecasts[0]['wind']['speed']
                            
                            precip_chance = 0
                            if any('rain' in f.get('weather', [{}])[0].get('main', '').lower() for f in day_forecasts):
                                precip_chance = 60
                            
                            temp_high = max(temps)
                            temp_low = min(temps)
                            main_condition = max(set(conditions), key=conditions.count)
                            
                            recommendation = generate_weather_recommendation(temp_high, temp_low, main_condition, precip_chance)
                            
                            daily_forecasts.append({
                                "date": date_str,
                                "temperature_high": round(temp_high, 1),
                                "temperature_low": round(temp_low, 1),
                                "condition": main_condition,
                                "humidity": humidity,
                                "wind_speed": wind_speed,
                                "precipitation_chance": precip_chance,
                                "recommendation": recommendation
                            })
                    return daily_forecasts
        return []
    except Exception as e:
        return []

def generate_weather_recommendation(temp_high: float, temp_low: float, condition: str, precip_chance: int) -> str:
    """Generate weather-based recommendations"""
    recommendations = []
    
    if temp_high > 25:
        recommendations.append("Pack light, breathable clothing")
    elif temp_high < 10:
        recommendations.append("Pack warm layers and winter gear")
    else:
        recommendations.append("Pack layers for varying temperatures")
    
    if precip_chance > 50:
        recommendations.append("Bring umbrella/rain gear")
    
    if "sun" in condition.lower():
        recommendations.append("Don't forget sunscreen and sunglasses")
    
    return "; ".join(recommendations)

@tool
def calculate_budget_breakdown(total_budget: float, days: int, members: int, hotel_price: float, transport_price: float) -> Dict:
    """Calculate detailed budget breakdown"""
    try:
        accommodation_total = hotel_price
        transport_total = transport_price
        
        remaining_budget = total_budget - accommodation_total - transport_total
        
        food_budget = remaining_budget * 0.4
        activities_budget = remaining_budget * 0.4
        emergency_fund = remaining_budget * 0.2
        
        daily_allowance = (food_budget + activities_budget) / days
        
        return {
            "total_budget": total_budget,
            "accommodation": accommodation_total,
            "transport": transport_total,
            "activities": activities_budget,
            "food": food_budget,
            "emergency": emergency_fund,
            "remaining": remaining_budget,
            "daily_allowance": daily_allowance,
            "per_person_daily": daily_allowance / members
        }
    except Exception as e:
        return {"error": f"Budget calculation failed: {str(e)}"}

# Initialize tools
tavily_tool = TavilySearchResults()

hotel_tools = [tavily_tool]
transport_tools = [tavily_tool]
activites_tools = [recommend_activities_tool, tavily_tool]
monitor_tools = [search_events, get_weather_forecast]
food_tools = [tavily_tool]
budget_tools = [calculate_budget_breakdown]

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful travel planner AI agent working in a supervised multi-agent system."
        " Your goal is to collaboratively create a structured, practical travel plan."
        " Always communicate clearly with other agents and follow supervisor guidance."
        " Use the available tools to get real data and provide accurate information."
        " Be efficient and complete your tasks in one iteration when possible."
        " Prefix your final response with FINAL ANSWER when your task is complete."
        f"\n{suffix}"
    )


# budget_tools, search_events, get_weather_forecast
context_budget_agent = create_react_agent(
    llm,
    tools=monitor_tools + budget_tools,
    prompt=make_system_prompt(
        "You are the CONTEXT & BUDGET MONITOR AGENT.\n"
        "- Fetch weather forecast and upcoming events between the trip dates.\n"
        "- Immediately calculate a realistic budget breakdown: stay, travel, food, activities, and emergencies.\n"
        "- Use tools to get real-time weather and events. Skip if no data found, but mark the task as done.\n"
        "- Avoid unnecessary reasoning or overthinking. Just extract and summarize the data.\n"
        "- FINAL ANSWER must include: weather forecast (at least 3 days), at least one event, and budget breakdown.\n"
    )

)

def context_budget_node(state: ExtendedMessagesState) -> Command:
    # Check if already completed
    if state.get("completed_tasks", {}).get("context_budget", False):
        return Command(update={"messages": [HumanMessage(content="FINAL ANSWER: Context and budget analysis already completed.", name="context_budget_monitor")]})
    
    result = context_budget_agent.invoke(state)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="context_budget_monitor"
    )
    
    # Mark as completed
    completed_tasks = state.get("completed_tasks", {})
    completed_tasks["context_budget"] = True
    
    return Command(update={
        "messages": result["messages"],
        "completed_tasks": completed_tasks
    })

# recommend_activities_tool
schedule_activity_agent = create_react_agent(
    llm,
    tools=activites_tools + [tavily_tool],
    prompt=make_system_prompt(
    "You are the SCHEDULE & ACTIVITY MANAGER AGENT.\n"
    "- Search for 6 to 9 relevant activities (free and paid) using available tools.\n"
    "- Distribute activities across 3 time blocks per day: Morning (9-12), Afternoon (1-5), Evening (6-9).\n"
    "- Avoid verbose justification. Just return a realistic and compact daily schedule.\n"
    "- Ensure no time block is overbooked.\n"
    "- FINAL ANSWER should contain: day-wise plan with 3 slots/day, activity name, location, and timing."
)

)

def schedule_activity_node(state: ExtendedMessagesState) -> Command:
    # Check if already completed
    if state.get("completed_tasks", {}).get("schedule_activity", False):
        return Command(update={"messages": [HumanMessage(content="FINAL ANSWER: Schedule and activities already planned.", name="schedule_activity_manager")]})
    
    result = schedule_activity_agent.invoke(state)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="schedule_activity_manager"
    )
    
    # Mark as completed
    completed_tasks = state.get("completed_tasks", {})
    completed_tasks["schedule_activity"] = True
    
    return Command(update={
        "messages": result["messages"],
        "completed_tasks": completed_tasks
    })


# hotels, transport, and food
logistics_agent = create_react_agent(
    llm,
    tools=hotel_tools + transport_tools + food_tools,
    prompt=make_system_prompt(
    "You are the LOGISTICS AGENT.\n"
    "- Quickly search for 2-3 hotels in different price ranges near the city center or activity locations.\n"
    "- Suggest at least 1 intercity transport option and 1 local transport method.\n"
    "- Search for 2-3 food places based on preferences and location.\n"
    "- Minimize reasoning or description — provide data-rich output (price, rating, address).\n"
    "- Mark task complete even if some info (e.g., food) is unavailable.\n"
    "- FINAL ANSWER must include: hotels, transport, and food suggestions, each in a bulleted format."
)

)

def logistics_node(state: ExtendedMessagesState) -> Command:
    # Check if already completed
    if state.get("completed_tasks", {}).get("logistics", False):
        return Command(update={"messages": [HumanMessage(content="FINAL ANSWER: Logistics already handled.", name="logistics")]})
    
    result = logistics_agent.invoke(state)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="logistics"
    )
    
    # Mark as completed
    completed_tasks = state.get("completed_tasks", {})
    completed_tasks["logistics"] = True
    
    return Command(update={
        "messages": result["messages"],
        "completed_tasks": completed_tasks
    })

# IMPROVED: Supervisor Agent with strict completion logic
supervisor_agent = create_react_agent(
    llm,
    tools=[],
    prompt=make_system_prompt(
    "You are the SUPERVISOR AGENT managing a travel planner system.\n"
    "- Your job is to check if all agents (context/budget, schedule/activity, logistics) are complete.\n"
    "- If yes, generate a clean, structured FINAL ANSWER with all data combined: schedule, logistics, weather, and budget.\n"
    "- Avoid requesting revision unless a key task is truly missing.\n"
    "- Prefer completion over perfection — deliver a usable plan to the user quickly.\n"
    "- If FINAL ANSWER already exists, do not continue.\n"
)

)

def supervisor_node(state: ExtendedMessagesState) -> Command:
    # Check iteration count to prevent infinite loops
    iteration_count = state.get("iteration_count", 0) + 1
    if iteration_count >= state.get("max_iterations", 5):
        final_summary = "FINAL ANSWER: Maximum iterations reached. Providing available travel plan based on current information."
        return Command(update={
            "messages": [HumanMessage(content=final_summary, name="supervisor")],
            "iteration_count": iteration_count
        })
    
    # Check if all tasks are completed
    completed_tasks = state.get("completed_tasks", {})
    all_completed = all([
        completed_tasks.get("context_budget", False),
        completed_tasks.get("schedule_activity", False),
        completed_tasks.get("logistics", False)
    ])
    
    if all_completed:
        # Generate final comprehensive summary
        supervisor_prompt = (
            "All agents have completed their tasks. Create a FINAL ANSWER with a comprehensive "
            "travel plan summary that integrates weather/budget info, daily schedule with activities, "
            "and logistics (hotels, transport, food). Make it ready for the user."
        )
    else:
        supervisor_prompt = (
            "Review the travel plan progress. If any critical information is missing, "
            "specify which agent needs to complete their task. Otherwise, provide FINAL ANSWER."
        )
    
    enhanced_messages = state["messages"] + [SystemMessage(content=supervisor_prompt)]
    result = supervisor_agent.invoke({"messages": enhanced_messages})
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="supervisor"
    )
    
    return Command(update={
        "messages": result["messages"],
        "iteration_count": iteration_count
    })

# IMPROVED: Workflow with reduced feedback loops
workflow = StateGraph(ExtendedMessagesState)

workflow.add_node("context_budget_monitor", context_budget_node)
workflow.add_node("schedule_activity_manager", schedule_activity_node)
workflow.add_node("logistics", logistics_node)
workflow.add_node("supervisor", supervisor_node)

# IMPROVED: Linear flow with minimal feedback
workflow.add_edge(START, "context_budget_monitor")
workflow.add_edge("context_budget_monitor", "schedule_activity_manager")
workflow.add_edge("schedule_activity_manager", "logistics")
workflow.add_edge("logistics", "supervisor")

# IMPROVED: Simplified supervisor routing - focus on completion
def supervisor_router(state: ExtendedMessagesState) -> str:
    last_message = state["messages"][-1].content.lower()
    
    # Always end if final answer is provided
    if "final answer" in last_message:
        return END
    
    # Check for explicit revision requests (rare)
    if "revision needed" in last_message or "revise" in last_message:
        if "weather" in last_message or "budget" in last_message:
            return "context_budget_monitor"
        elif "activity" in last_message or "schedule" in last_message:
            return "schedule_activity_manager"
        elif "hotel" in last_message or "transport" in last_message or "food" in last_message:
            return "logistics"
    
    # Default to ending to prevent loops
    return END

workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    ["context_budget_monitor", "schedule_activity_manager", "logistics", END]
)

graph = workflow.compile()

def run_travel_planner():
    """Main function to run the enhanced travel planning system"""
    # Collect user input
    destination = "ladakh"
    days = 7
    budget = 10000
    current_location = "vijayawada"
    food_pref = "indian"
    members = 3
    start_date = "2025-7-24"
    
    # Calculate end date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = (start_dt + timedelta(days=days-1)).strftime("%Y-%m-%d")
    
    # Build comprehensive user message
    user_message = (
        f"Plan a comprehensive {days}-day trip from {current_location} to {destination} "
        f"from {start_date} to {end_date} for {members} members with a total budget of ${budget}. "
        f"Food preference: {food_pref}. "
        f"Include: detailed daily schedule, weather-appropriate activities, hotel options with "
        f"images/ratings/prices, transport options with timings, food recommendations with local cuisine, "
        f"budget breakdown, and local events. "
        f"Ensure everything is realistic, within budget, and considers weather conditions."
    )
    
    try:
        start_time = time.time()
        
        # Initialize state with completion tracking
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "completed_tasks": {},
            "iteration_count": 0,
            "max_iterations": 5
        }
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        processing_time = time.time() - start_time
        
        # Extract and display the final plan
        final_plan_found = False
        for msg in reversed(final_state["messages"]):  # Check from most recent
            if hasattr(msg, 'name') and msg.name == "supervisor" and "FINAL ANSWER" in msg.content:
                final_plan = msg.content.replace("FINAL ANSWER", "").strip()
                print("🌟 COMPREHENSIVE TRAVEL PLAN 🌟")
                print("=" * 50)
                print(f"📍 Destination: {destination.upper()}")
                print(f"📅 Duration: {days} days ({start_date} to {end_date})")
                print(f"👥 Travelers: {members} members")
                print(f"💰 Budget: ${budget:,.2f}")
                print(f"🍽️ Food Preference: {food_pref}")
                print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
                print(f"🔄 Iterations: {final_state.get('iteration_count', 0)}")
                print("=" * 50)
                print()
                print(final_plan)
                final_plan_found = True
                break
        
        if not final_plan_found:
            print("🌟 COMPREHENSIVE TRAVEL PLAN 🌟")
            print("=" * 50)
            print(f"📍 Destination: {destination.upper()}")
            print(f"📅 Duration: {days} days ({start_date} to {end_date})")
            print(f"👥 Travelers: {members} members")
            print(f"💰 Budget: ${budget:,.2f}")
            print(f"🍽️ Food Preference: {food_pref}")
            print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
            print(f"🔄 Iterations: {final_state.get('iteration_count', 0)}")
            print("=" * 50)
            print()
            # Get the last meaningful message
            for msg in reversed(final_state["messages"]):
                if msg.content.strip() and not msg.content.startswith("FINAL ANSWER"):
                    print(msg.content.strip())
                    break
        
    except Exception as e:
        print("❌ ERROR OCCURRED")
        print("=" * 20)
        print(f"Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your API keys are correctly set")
        print("2. Ensure internet connection is stable")
        print("3. Verify date format is YYYY-MM-DD")
        print("4. Try with a different destination if issue persists")

# if __name__ == "__main__":
#     print("🌟 ENHANCED MULTI-AGENT TRAVEL PLANNING SYSTEM 🌟")
#     print("✅ OPTIMIZED FOR FEWER ITERATIONS")
#     print("Features: Real-time weather, local events, budget tracking,")
#     print("hotel search, transport options, food recommendations, and supervised AI collaboration")
#     print("=" * 70)
    
run_travel_planner()
