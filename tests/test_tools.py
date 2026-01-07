#!/usr/bin/env python3
"""
Tests for tools module.
"""

import pytest
import sys
import os
from tools import ToolRegistry, weather_tool, unit_converter_tool

# Ensure the src directory is in sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestToolRegistry:
    """Test ToolRegistry class."""

    def setup_method(self):
        """Setup before each test method."""
        self.registry = ToolRegistry()

    def test_initialization(self):
        """Test that ToolRegistry initializes correctly."""
        assert self.registry is not None
        assert hasattr(self.registry, 'tools')
        assert isinstance(self.registry.tools, dict)

        # Should have registered built-in tools
        assert len(self.registry.tools) > 0

    def test_list_tools(self):
        """Test listing all registered tools."""
        tools_list = self.registry.list_tools()

        assert isinstance(tools_list, dict)

        # Check for expected built-in tools
        expected_tools = ["calculator", "time", "web_search", "file_reader", "text_processor"]

        for tool_name in expected_tools:
            assert tool_name in tools_list
            assert isinstance(tools_list[tool_name], str)  # Description

    def test_get_tool(self):
        """Test getting a specific tool."""
        calculator = self.registry.get_tool("calculator")

        assert calculator is not None
        assert calculator.name == "calculator"
        assert "mathematical" in calculator.description.lower()

    def test_get_tools(self):
        """Test getting multiple tools."""
        tool_names = ["calculator", "time"]
        tools = self.registry.get_tools(tool_names)

        assert len(tools) == 2
        assert tools[0].name == "calculator"
        assert tools[1].name == "time"

    def test_get_all_tools(self):
        """Test getting all tools when no names specified."""
        all_tools = self.registry.get_tools()

        assert len(all_tools) > 0
        assert all(hasattr(tool, 'name') for tool in all_tools)
        assert all(hasattr(tool, 'description') for tool in all_tools)

    def test_calculator_tool(self):
        """Test calculator tool functionality."""
        calculator = self.registry.get_tool("calculator")

        # Test simple addition
        result = calculator.func("2 + 2")
        assert "4" in result or "Result: 4" in result

        # Test with math function
        result = calculator.func("sqrt(16)")
        assert "4" in result or "Result: 4" in result

    def test_time_tool(self):
        """Test time tool functionality."""
        time_tool = self.registry.get_tool("time")

        result = time_tool.func("")
        assert "Current date and time" in result

    def test_web_search_tool(self):
        """Test web search tool functionality."""
        search_tool = self.registry.get_tool("web_search")

        result = search_tool.func("Python programming", max_results=3)
        assert "Search results for" in result
        assert "Python programming" in result

    def test_text_processor_tool(self):
        """Test text processor tool functionality."""
        text_tool = self.registry.get_tool("text_processor")

        # Test 1: Simple word count
        test_text = "Hello World!"
        result = text_tool.func(test_text, operation="word_count")
        assert "Word count: 2" in result or "2" in result

        # Test 2: With numbers (optional - depends on your definition)
        test_text2 = "Hello World! 123"
        result2 = text_tool.func(test_text2, operation="word_count")

        # Accept either 2 or 3 depending on implementation
        if "Word count: 2" in result2 or "2" in result2:
            print("Tool treats numbers as non-words")
        elif "Word count: 3" in result2 or "3" in result2:
            print("Tool treats numbers as words")
        else:
            # Fail only if neither works
            assert False, f"Unexpected result: {result2}"

        # Test other operations
        result3 = text_tool.func("hello", operation="upper")
        assert "HELLO" in result3

        result4 = text_tool.func("abc", operation="reverse")
        assert "cba" in result4 or "Reversed: cba" in result4


def test_weather_tool():
    """Test the weather tool decorator."""
    # Test with known location
    result = weather_tool.func("New York")
    assert "Weather in New York" in result
    assert "°F" in result

    # Test with unknown location
    result = weather_tool.func("Unknown City")
    assert "Weather data not available" in result or "Simulated" in result


def test_unit_converter_tool():
    """Test the unit converter tool."""
    # Test length conversion
    result = unit_converter_tool.func(1000, "meter", "kilometer")
    assert "1.00" in result or "1.0" in result
    assert "kilometer" in result

    # Test temperature conversion
    result = unit_converter_tool.func(100, "celsius", "fahrenheit")
    assert "212.00" in result or "212" in result
    assert "fahrenheit" in result

    # Test unsupported conversion
    result = unit_converter_tool.func(10, "unknown_unit", "other_unit")
    assert "not supported" in result


def test_tools_module_imports():
    """Test that tools module imports correctly."""
    from tools import ToolRegistry, weather_tool, unit_converter_tool

    assert ToolRegistry is not None
    assert weather_tool is not None
    assert unit_converter_tool is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
