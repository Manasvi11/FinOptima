#import streamlit as st
from phi.agent import Agent 
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os

from dotenv import load_dotenv
load_dotenv()
Groq.api_key=os.getenv("GROQ_API_KEY")
#GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
#phi_API_KEY=os.getenv("phi_API_KEY")
#phi_API_KEY="phi-GG-ln-bQEYFvcHD9wD3xZF5rrfAHdXU_n2ThBAQahyY"
#GOOGLE_API_KEY="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"
#os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
#os.environ["phi_API_KEY"]=phi_API_KEY

search_wed_agent=Agent(
  model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  
  name="search wed agent",
  role="Searh the web for information,articles,news,etc",
  tools=[DuckDuckGo()],
  instructions=["Always inlude sources"],
  show_tool_calls=True,
  markdown=True,
 )


finance_agent=Agent(
  model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  
  name="finance  agent",
  tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
  description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
  instructions=["use tables,graphs,charts to display data where possible."],
  show_tool_calls=True,
  markdown=True,
 )

multi_agent=Agent( 
    team=[search_wed_agent,finance_agent],
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    instructions=["Always inlude sources ","use tables,graphs,charts to display data where possible"],
    show_tool_calls=True,
    markdown=True,
)

multi_agent.print_response("Share the NVIDAI analyst recomendation and latest news or any articles about NVIDIA", stream=True)


