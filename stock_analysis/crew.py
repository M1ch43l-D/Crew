import os
import dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from crewai import Crew
from textwrap import dedent
from stock_analysis.stock_analysis_agents import StockAnalysisAgents
from stock_analysis.stock_analysis_tasks import StockAnalysisTasks
from dotenv import load_dotenv
load_dotenv()
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

researcher = Agent(
  role='Staff Research Analyst',
      goal="""Being the best at gather, interpret data and amaze
      your customer with it""",
      backstory="""Known as the BEST research analyst, you're
      skilled in sifting through news, company announcements, 
      and market sentiments. Now you're working on a super 
      important customer""",
      verbose=True,
  allow_delegation=False,
  tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        SearchTools.search_news,
        YahooFinanceNewsTool()
      ]
)

writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
)

task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in Cybersecurity and AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant Cybersecurity and AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)