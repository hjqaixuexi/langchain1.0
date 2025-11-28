from calendar_agent import calendar_agent  # ty: ignore
from dotenv import load_dotenv
from email_agent import email_agent  # ty: ignore
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


supervisor_agent = create_agent(
    model="deepseek:deepseek-chat",
    system_prompt=SUPERVISOR_PROMPT,
    tools=[schedule_event, manage_email],
)


def test_supervisor_agent():
    query = "hello"
    for event in supervisor_agent.stream(
        {"messages": {"role": "user", "content": query}}, stream_mode="values"
    ):
        event["messages"][-1].pretty_print()

    query = "Schedule a team meeting ['design-team@abc.com'] on 2025-11-15 at 2pm 1 hour, then send email to notify the team"
    for event in supervisor_agent.stream(
        {"messages": {"role": "user", "content": query}}, stream_mode="values"
    ):
        event["messages"][-1].pretty_print()


# test_supervisor_agent()
