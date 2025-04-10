import asyncio
from autogen import GroupChat, GroupChatManager, UserProxyAgent, AssistantAgent
from autogen_ext.tools.mcp import mcp_ws_client_tools, WebSocketClientParams

# ───────────────────────────────────────────────────────────────
# 🔌 Connect to MCP WebSocket Server
# ───────────────────────────────────────────────────────────────

async def create_mcp_tools():
    # Assumes the MCP server is running at this address inside the container
    params = WebSocketClientParams(url="ws://localhost:8090")
    return await mcp_ws_client_tools(params)


# ───────────────────────────────────────────────────────────────
# 🛫 Booking Agent
# ───────────────────────────────────────────────────────────────

def create_booking_agent(tools):
    return AssistantAgent(
        name="BookingAgent",
        llm_config={"config_list": [{"model": "gpt-4"}]},
        tools=tools,
        system_message="""
            You are a flight booking agent.
            Book flights using the provided origin, destination, and date.
            Save the booking details (origin, destination, date, booking ID) using the MCP tools.
        """,
    )


# ───────────────────────────────────────────────────────────────
# 💺 Seat Selection Agent
# ───────────────────────────────────────────────────────────────

def create_seat_selection_agent(tools):
    return AssistantAgent(
        name="SeatSelectionAgent",
        llm_config={"config_list": [{"model": "gpt-4"}]},
        tools=tools,
        system_message="""
            You are a seat selection agent.
            Retrieve booking information using the tools.
            Assign an aisle seat and persist the seat selection.
        """,
    )


# ───────────────────────────────────────────────────────────────
# 🧭 Triage Agent
# ───────────────────────────────────────────────────────────────

def create_triage_agent():
    return UserProxyAgent(
        name="TriageAgent",
        human_input_mode="NEVER",
        code_execution_config=False,
        system_message="""
            You triage user travel requests.
            - If the request involves flight booking, delegate to BookingAgent.
            - If the request involves seat selection, delegate to SeatSelectionAgent.
            Guide the workflow and ensure all tasks are completed.
        """,
    )


# ───────────────────────────────────────────────────────────────
# 💬 Group Chat Configuration
# ───────────────────────────────────────────────────────────────

def create_group_chat(agents):
    return GroupChat(
        agents=agents,
        messages=[],
        max_round=8,
    )

def create_group_chat_manager(group_chat):
    return GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": [{"model": "gpt-4"}]},
    )


# ───────────────────────────────────────────────────────────────
# 🚀 Entry Point
# ───────────────────────────────────────────────────────────────

async def main():
    # Connect to the MCP WebSocket server running inside Codespaces
    tools = await create_mcp_tools()

    # Create agents with shared MCP tools
    booking_agent = create_booking_agent(tools)
    seat_agent = create_seat_selection_agent(tools)
    triage_agent = create_triage_agent()

    # Group coordination
    group_chat = create_group_chat([triage_agent, booking_agent, seat_agent])
    manager = create_group_chat_manager(group_chat)

    # Kick off the workflow
    triage_agent.initiate_chat(
        manager,
        message="I want to book a flight from New York to London on April 20, and I prefer an aisle seat.",
    )


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
