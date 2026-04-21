"""Entry point for running the TrustRAG MCP server via `python -m trustrag_mcp`."""

import asyncio

from mcp.server.stdio import stdio_server

from trustrag_mcp.server import app


def main():
    async def run():
        async with stdio_server() as streams:
            await app.run(
                streams[0],
                streams[1],
                app.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
