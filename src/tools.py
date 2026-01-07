"""
Tools module for defining custom tools that can be used by LangChain agents.
"""

from typing import Optional, Dict, List
from langchain_core.tools import BaseTool, Tool
from langchain_core.tools import tool as langchain_tool
from datetime import datetime
import math


class ToolRegistry:
    """Registry for managing and organizing tools."""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register built-in tools."""
        self.register_tool(self.create_calculator_tool())
        self.register_tool(self.create_time_tool())
        self.register_tool(self.create_web_search_tool())
        self.register_tool(self.create_file_reader_tool())
        self.register_tool(self.create_text_processing_tool())

    def register_tool(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: LangChain tool to register
        """
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)

    def get_tools(self, names: Optional[List[str]] = None) -> List[BaseTool]:
        """
        Get list of tools.

        Args:
            names: Optional list of tool names to get

        Returns:
            List of tools
        """
        if names is None:
            return list(self.tools.values())

        return [self.tools[name] for name in names if name in self.tools]

    def list_tools(self) -> Dict[str, str]:
        """
        List all registered tools with descriptions.

        Returns:
            Dictionary of tool names and descriptions
        """
        return {
            name: tool.description
            for name, tool in self.tools.items()
        }

    def create_calculator_tool(self) -> BaseTool:
        """Create a calculator tool."""

        def calculate(expression: str) -> str:
            """
            Evaluate a mathematical expression.

            Args:
                expression: Mathematical expression to evaluate

            Returns:
                Result of the calculation
            """
            try:
                # Security: Only allow safe operations
                allowed_names = {
                    k: v for k, v in math.__dict__.items()
                    if not k.startswith("_")
                }
                allowed_names.update({
                    "abs": abs,
                    "round": round,
                    "min": min,
                    "max": max,
                    "sum": sum,
                })

                # Evaluate expression
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return f"Result: {result}"
            except Exception as e:
                return f"Error calculating expression: {e}"

        return Tool(
            name="calculator",
            func=calculate,
            description="""Useful for performing mathematical calculations.
            Input should be a mathematical expression like '2 + 2' or 'sqrt(16)'."""
        )

    def create_time_tool(self) -> BaseTool:
        """Create a tool to get current time and date."""

        def get_current_time(timezone: Optional[str] = None) -> str:
            """
            Get current time and date.

            Args:
                timezone: Optional timezone (not implemented in basic version)

            Returns:
                Current time and date string
            """
            now = datetime.now()
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

        return Tool(
            name="time",
            func=get_current_time,
            description="""Useful for getting the current date and time.
            Input can be empty or specify 'timezone' parameter."""
        )

    def create_web_search_tool(self) -> BaseTool:
        """Create a web search tool (simulated)."""

        def web_search(query: str, max_results: int = 5) -> str:
            """
            Search the web for information.

            Args:
                query: Search query
                max_results: Maximum number of results to return

            Returns:
                Search results as string
            """
            # Note: This is a simulated search. In production, you would integrate
            # with a real search API like Google Search or DuckDuckGo.

            simulated_results = [
                f"Result 1: Information about {query} - This would be real search result data.",
                f"Result 2: More details on {query} from reputable sources.",
                f"Result 3: Recent developments related to {query}.",
                f"Result 4: Tutorials and guides for {query}.",
                f"Result 5: Community discussions about {query}.",
            ]

            return f"Search results for '{query}':\n" + "\n".join(
                simulated_results[:max_results]
            )

        return Tool(
            name="web_search",
            func=web_search,
            description="""Useful for searching the web for current information.
            Input should be a search query string."""
        )

    def create_file_reader_tool(self) -> BaseTool:
        """Create a tool to read files (simulated for security)."""

        def read_file(filepath: str, max_lines: int = 100) -> str:
            """
            Read contents of a file.

            Args:
                filepath: Path to the file
                max_lines: Maximum number of lines to read

            Returns:
                File contents as string
            """
            # Security note: In production, this should have proper security controls
            # and only allow reading from specific directories.

            try:
                # Simulated file reading
                if "secret" in filepath.lower():
                    return "Access denied: Cannot read secret files."

                simulated_content = f"Simulated content of {filepath}:\n"
                simulated_content += f"This is line 1 of {filepath}\n"
                simulated_content += "This is line 2 with more details\n"
                simulated_content += f"... up to {max_lines} lines would be shown here."

                return simulated_content
            except Exception as e:
                return f"Error reading file: {e}"

        return Tool(
            name="file_reader",
            func=read_file,
            description="Useful for reading text files. Input should be a file path."
        )

    def create_text_processing_tool(self) -> BaseTool:
        """Create a tool for text processing operations."""

        def process_text(
            text: str,
            operation: str = "word_count",
            **kwargs
        ) -> str:
            """
            Process text with various operations.

            Args:
                text: Text to process
                operation: Type of operation (word_count, character_count, reverse, upper, lower)
                **kwargs: Additional operation-specific parameters

            Returns:
                Processed text or analysis result
            """
            operation = operation.lower()

            if operation == "word_count":
                words = text.split()
                return f"Word count: {len(words)}"

            elif operation == "character_count":
                return f"Character count: {len(text)}"

            elif operation == "reverse":
                return f"Reversed text: {text[::-1]}"

            elif operation == "upper":
                return f"Uppercase: {text.upper()}"

            elif operation == "lower":
                return f"Lowercase: {text.lower()}"

            elif operation == "extract_numbers":
                import re
                numbers = re.findall(r'\d+\.?\d*', text)
                return f"Numbers found: {', '.join(numbers)}"

            else:
                return f"""Unknown operation: {operation}.
                Available: word_count, character_count, reverse, upper, lower, extract_numbers"""

        return Tool(
            name="text_processor",
            func=process_text,
            description="""Useful for text processing operations.
            Input should include text and operation type."""
        )


@langchain_tool
def weather_tool(location: str) -> str:
    """
    Get current weather for a location.

    Args:
        location: City name or location

    Returns:
        Weather information
    """
    # Note: This is a simulated weather tool
    # In production, integrate with a real weather API

    simulated_weather = {
        "New York": "75°F, Sunny",
        "London": "60°F, Cloudy",
        "Tokyo": "68°F, Partly Cloudy",
        "Sydney": "72°F, Clear",
    }

    if location in simulated_weather:
        return f"Weather in {location}: {simulated_weather[location]}"
    else:
        return f"Weather data not available for {location}. Simulated: 70°F, Mostly Sunny"


@langchain_tool
def unit_converter_tool(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """
    Convert between different units.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Conversion result
    """
    # Define conversion rates
    conversions = {
        # Length
        ("meter", "kilometer"): lambda x: x / 1000,
        ("kilometer", "meter"): lambda x: x * 1000,
        ("mile", "kilometer"): lambda x: x * 1.60934,
        ("kilometer", "mile"): lambda x: x / 1.60934,

        # Weight
        ("kilogram", "pound"): lambda x: x * 2.20462,
        ("pound", "kilogram"): lambda x: x / 2.20462,

        # Temperature
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,

        # Currency (simulated rates)
        ("usd", "eur"): lambda x: x * 0.92,
        ("eur", "usd"): lambda x: x / 0.92,
    }

    key = (from_unit.lower(), to_unit.lower())
    reverse_key = (to_unit.lower(), from_unit.lower())

    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    elif reverse_key in conversions:
        result = 1 / conversions[reverse_key](1/value) if value != 0 else 0
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported."


class CustomTool(BaseTool):
    """Base class for custom tools."""

    def _run(self, *args, **kwargs):
        raise NotImplementedError

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError


class WikipediaSearchTool(CustomTool):
    """Tool for searching Wikipedia (simulated)."""

    name: str = "wikipedia_search"
    description: str = "Search Wikipedia for information on a topic."

    def _run(self, query: str) -> str:
        """Search Wikipedia."""
        # Simulated Wikipedia search
        return f"Wikipedia results for '{query}':\n" \
               f"1. {query} - Overview and history\n" \
               f"2. Applications of {query}\n" \
               f"3. Recent developments in {query} field\n" \
               f"Note: This is simulated data. Integrate with Wikipedia API for real results."

    async def _arun(self, query: str) -> str:
        """Async version of Wikipedia search."""
        return self._run(query)
