from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("Chat bot")

print("Mcp>>>",mcp.name)

@mcp.tool()
def chatbot(query:str)->str:
    """
    Generate a answer for user asked query
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a AI assistant will answer for users questions"
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ == "__main__":
    mcp.run(transport="stdio")