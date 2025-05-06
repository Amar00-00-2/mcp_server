from typing import Optional
from contextlib import AsyncExitStack
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from utils.logger import logger
import json
import os
from openai import OpenAI

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()
        self.tools = []
        self.messages = []
        self.logger = logger

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            self.logger.info("Connected to MCP server")

            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "type": "function",  # Add this required field
                    "function": {  # Wrap properties in a function object
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,  # Rename input_schema to parameters
                    }
                }
                for tool in mcp_tools
            ]

            self.logger.info(
                f"Available tools: {[tool['function']['name'] for tool in self.tools]}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            user_message = {"role": "user", "content": query}
            self.messages = [user_message]

            while True:
                response = await self.call_llm()

                # the response is a text message
                if not hasattr(response, 'choices') or not response.choices[0].message.tool_calls:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                # the response is a tool call
                message = response.choices[0].message
                assistant_message = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                }
                self.messages.append(assistant_message)
                await self.log_conversation()

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_use_id = tool_call.id
                    self.logger.info(
                        f"Calling tool {tool_name} with args {tool_args}"
                    )
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        self.logger.info(f"Tool {tool_name} result: {result}...")
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": result.content,
                            }
                        )
                        await self.log_conversation()
                    except Exception as e:
                        self.logger.error(f"Error calling tool {tool_name}: {e}")
                        raise

            return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            # return self.llm.messages.create(
            #     model="claude-3-5-haiku-20241022",
                # max_tokens=1000,
                # messages=self.messages,
                # tools=self.tools,
            # )
            return self.llm.chat.completions.create(
                model="gpt-4o",
                max_tokens=1000,
                messages=self.messages,
                tools=self.tools,
            )
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"]}
                
                # Handle content
                if "content" in message:
                    serializable_message["content"] = message["content"]
                
                # Handle tool_calls if present
                if "tool_calls" in message:
                    serializable_message["tool_calls"] = []
                    for tool_call in message["tool_calls"]:
                        if hasattr(tool_call, "to_dict"):
                            serializable_message["tool_calls"].append(tool_call.to_dict())
                        elif isinstance(tool_call, dict):
                            serializable_message["tool_calls"].append(tool_call)
                        else:
                            serializable_message["tool_calls"].append({
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            })
                
                # Handle tool_call_id if present (for tool responses)
                if "tool_call_id" in message:
                    serializable_message["tool_call_id"] = message["tool_call_id"]

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise