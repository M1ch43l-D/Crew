import os
from crewai import Agent, Task, Crew
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
load_dotenv()

#assert os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "Null"
openai_api_base = "http://localhost:1234/v1"
llm = ChatOpenAI(model_name="gpt-3.5", temperature=0.7, openai_api_base=openai_api_base)

search_tool = DuckDuckGoSearchRun()

researcher = Agent(
    role='Senior Research Analyst',
    goal='Conduct research on given topics, focusing on data accuracy and comprehensive analysis.',
    backstory="""As a world-renowned research analyst, your expertise lies in diving deep into topics, unraveling layers of information to uncover facts and data. Your research is not just surface-level; you delve into multiple iterations to ensure no stone is left unturned, always backing your findings with solid evidence.""",
    verbose=True,
    allow_delegation=False,
    tools=[
        BrowserTools.scrape_and_summarize_website,
        #SearchTools.search_internet,
        search_tool,
        SearchTools.search_news,
        YahooFinanceNewsTool()
      ],
      llm=llm
)

strategist = Agent(
    role='Senior Strategist',
    goal='Transform complex ideas into structured, actionable plans, focusing on foundational understanding and innovative problem-solving.',
    backstory="""Your reputation as a senior strategist stems from your ability to dissect complex ideas, finding the underlying principles that others miss. Your approach is not conventional; its about rebuilding ideas from the ground up, ensuring each solution is not just effective but also groundbreaking.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# saver = Agent(
#     role='Content Saver',
#     goal='Save compelling content on tech advancements',
#     backstory="""You are a renowned Content Strategist, known for
#   your insightful and engaging articles.
#   You transform complex concepts into compelling narratives.""",
#     verbose=True,
#     allow_delegation=True,
#     llm=llm
# )

task1 = Task(
    description="""Research David Ogilvy. and his gratest works headlines and sales letters.""",
    agent=researcher
)

task2 = Task(
    description="""Using the research provided. craft a sales letter inviting creative entreprenures to attend our book club. The book title is the creative act by rick ruben. the gratest hiphop producer of all time.""",
    agent=strategist
)

crew = Crew(
    agents=[researcher, strategist],
    tasks=[task1, task2],
    verbose=2, 
)

result = crew.kickoff()
print("######################")
print(result)