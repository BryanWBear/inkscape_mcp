#!/usr/bin/env python3
"""
SVG Logo Design Agent using LangGraph and Claude
Creates minimal SVG logo designs using cubic Bezier curves
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass

import xml.etree.ElementTree as ET
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.errors import NodeInterrupt
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurvePoint:
    """Represents a 2D point for curve endpoints and control points"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass
class CubicBezierCurve:
    """Represents a cubic Bezier curve with start, end, and control points"""
    start: CurvePoint
    end: CurvePoint
    control1: CurvePoint
    control2: CurvePoint
    stroke_color: str = "#000000"
    stroke_width: float = 2.0
    fill: str = "none"
    closed: bool = False


class CurveEndpoints(BaseModel):
    start_x: int = Field(description="Starting x coordinate of the curve")
    end_x: int = Field(description="Ending x coordinate of the curve")
    start_y: int = Field(description="Starting y coordinate of the curve")
    end_y: int = Field(description="Ending y coordinate of the curve")

class DesignPlan(BaseModel):
    """Pydantic model for the design plan"""
    curves: Dict[str, CurveEndpoints] = Field(description="List of curves to draw")
    canvas_width: int = Field(default=800, description="Canvas width")
    canvas_height: int = Field(default=600, description="Canvas height")
    description: str = Field(description="Description of the design")

class AgentState(BaseModel):
    """State for the LangGraph agent"""
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)
    user_request: str = ""
    design_plan: Optional[DesignPlan] = None
    current_curve_index: int = 0
    canvas_created: bool = False
    design_complete: bool = False
    svg_content: str = ""
    filename: str = "logo_design.svg"

class SVGCanvas:
    """Manages SVG canvas and drawing operations"""
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.curves = []
        self.curve_map = {}  # Map curve IDs to curve objects
        self.svg_root = self._create_svg_root()
    
    def _create_svg_root(self) -> ET.Element:
        """Create the root SVG element"""
        root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(self.width),
            "height": str(self.height),
            "viewBox": f"0 0 {self.width} {self.height}"
        })
        return root
    
    def add_cubic_bezier_curve(self, curve: CubicBezierCurve) -> str:
        """Add a cubic Bezier curve to the canvas"""
        path_id = f"curve_{len(self.curves)}"
        
        # Create SVG path data
        path_data = (
            f"M {curve.start.x},{curve.start.y} "
            f"C {curve.control1.x},{curve.control1.y} "
            f"{curve.control2.x},{curve.control2.y} "
            f"{curve.end.x},{curve.end.y}"
        )
        
        if curve.closed:
            path_data += " Z"
        
        # Create path element
        path_elem = ET.SubElement(self.svg_root, "path", {
            "id": path_id,
            "d": path_data,
            "stroke": curve.stroke_color,
            "stroke-width": str(curve.stroke_width),
            "fill": curve.fill
        })
        
        self.curves.append(curve)
        self.curve_map[path_id] = curve
        return path_id
    
    def modify_curve(self, curve_id: str, new_curve: CubicBezierCurve) -> bool:
        """Modify an existing curve by its ID"""
        if curve_id not in self.curve_map:
            return False
        
        # Update the curve in our map
        self.curve_map[curve_id] = new_curve
        
        # Find and update the corresponding SVG path element
        for path_elem in self.svg_root.findall(".//path[@id='{}']".format(curve_id)):
            # Update SVG path data
            path_data = (
                f"M {new_curve.start.x},{new_curve.start.y} "
                f"C {new_curve.control1.x},{new_curve.control1.y} "
                f"{new_curve.control2.x},{new_curve.control2.y} "
                f"{new_curve.end.x},{new_curve.end.y}"
            )
            
            if new_curve.closed:
                path_data += " Z"
            
            # Update path attributes
            path_elem.set("d", path_data)
            path_elem.set("stroke", new_curve.stroke_color)
            path_elem.set("stroke-width", str(new_curve.stroke_width))
            path_elem.set("fill", new_curve.fill)
            
            # Also update the curve in the curves list
            for i, curve in enumerate(self.curves):
                if self.curve_map.get(f"curve_{i}") == curve:
                    self.curves[i] = new_curve
                    break
            
            return True
        
        return False
    
    def get_curve_info(self, curve_id: str) -> Optional[CubicBezierCurve]:
        """Get information about a specific curve"""
        return self.curve_map.get(curve_id)
    
    def list_curves(self) -> Dict[str, str]:
        """List all curves with their IDs and basic info"""
        curve_info = {}
        for curve_id, curve in self.curve_map.items():
            curve_info[curve_id] = f"Start: ({curve.start.x}, {curve.start.y}), End: ({curve.end.x}, {curve.end.y})"
        return curve_info
    
    def to_svg_string(self) -> str:
        """Convert canvas to SVG string"""
        return ET.tostring(self.svg_root, encoding='unicode')
    
    def save_to_file(self, filename: str):
        """Save canvas to SVG file"""
        svg_string = self.to_svg_string()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_string)
        logger.info(f"SVG saved to {filename}")

# Global canvas instance
canvas = SVGCanvas()


@tool
def create_canvas(width: int = 800, height: int = 600, filename: str = "logo_design.svg") -> str:
    """
    Create a new SVG canvas with specified dimensions.
    
    Args:
        width: Canvas width in pixels
        height: Canvas height in pixels
        filename: Output filename for the SVG
        
    Returns:
        Success message
    """
    global canvas
    canvas = SVGCanvas(width, height)
    return f"Canvas created with dimensions {width}x{height}"

@tool
def add_cubic_bezier_curve(
    start_x: float, start_y: float,
    end_x: float, end_y: float,
    control1_x: float, control1_y: float,
    control2_x: float, control2_y: float,
    stroke_color: str = "#000000",
    stroke_width: float = 2.0,
    fill: str = "none",
    closed: bool = False
) -> str:
    """
    Add a cubic Bezier curve to the canvas.
    
    Args:
        start_x, start_y: Starting point coordinates
        end_x, end_y: Ending point coordinates
        control1_x, control1_y: First control point coordinates
        control2_x, control2_y: Second control point coordinates
        stroke_color: Hex color code for the stroke
        stroke_width: Width of the stroke line
        fill: Fill color or 'none'
        closed: Whether to close the path
        
    Returns:
        Path ID of the created curve
    """
    curve = CubicBezierCurve(
        start=CurvePoint(start_x, start_y),
        end=CurvePoint(end_x, end_y),
        control1=CurvePoint(control1_x, control1_y),
        control2=CurvePoint(control2_x, control2_y),
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill=fill,
        closed=closed
    )
    
    path_id = canvas.add_cubic_bezier_curve(curve)
    return f"Added curve {path_id}"

@tool
def modify_cubic_bezier_curve(
    curve_id: str,
    start_x: float, start_y: float,
    end_x: float, end_y: float,
    control1_x: float, control1_y: float,
    control2_x: float, control2_y: float,
    stroke_color: str = "#000000",
    stroke_width: float = 2.0,
    fill: str = "none",
    closed: bool = False
) -> str:
    """
    Modify an existing cubic Bezier curve by its unique ID.
    
    Args:
        curve_id: Unique ID of the curve to modify (e.g., "curve_0", "curve_1")
        start_x, start_y: New starting point coordinates
        end_x, end_y: New ending point coordinates
        control1_x, control1_y: New first control point coordinates
        control2_x, control2_y: New second control point coordinates
        stroke_color: New hex color code for the stroke
        stroke_width: New width of the stroke line
        fill: New fill color or 'none'
        closed: Whether to close the path
        
    Returns:
        Success or error message
    """
    # Check if curve exists
    if curve_id not in canvas.curve_map:
        available_curves = list(canvas.curve_map.keys())
        return f"Error: Curve '{curve_id}' not found. Available curves: {available_curves}"
    
    # Create new curve object
    new_curve = CubicBezierCurve(
        start=CurvePoint(start_x, start_y),
        end=CurvePoint(end_x, end_y),
        control1=CurvePoint(control1_x, control1_y),
        control2=CurvePoint(control2_x, control2_y),
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill=fill,
        closed=closed
    )
    
    # Modify the curve
    success = canvas.modify_curve(curve_id, new_curve)
    
    if success:
        return f"Successfully modified curve {curve_id}"
    else:
        return f"Error: Failed to modify curve {curve_id}"

@tool
def list_all_curves() -> str:
    """
    List all curves currently on the canvas with their IDs and basic information.
    
    Returns:
        Formatted string with curve information
    """
    curve_info = canvas.list_curves()
    
    if not curve_info:
        return "No curves found on canvas"
    
    result = "Current curves on canvas:\n"
    for curve_id, info in curve_info.items():
        curve = canvas.get_curve_info(curve_id)
        result += f"- {curve_id}: {info}"
        if curve:
            result += f", Stroke: {curve.stroke_color}, Width: {curve.stroke_width}, Fill: {curve.fill}"
        result += "\n"
    
    return result

@tool
def save_and_display_svg(filename: str = "logo_design.svg") -> str:
    """
    Save the current canvas to an SVG file and return the SVG content.
    
    Args:
        filename: Output filename for the SVG
        
    Returns:
        SVG content as string
    """
    canvas.save_to_file(filename)
    svg_content = canvas.to_svg_string()
    return f"SVG saved to {filename}. Content:\n{svg_content}"

# Initialize Claude

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('ANTHROPIC_API_KEY')

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0.1,
    api_key=api_key
)


# Tools available to the agent
tools = [create_canvas, add_cubic_bezier_curve, modify_cubic_bezier_curve, list_all_curves, save_and_display_svg]
tool_executor = ToolNode(tools)

system_prompt = """
You are a specialized React agent focused on creating SVG logo designs using cubic Bezier curves. 
Your primary goal is to transform user requests into elegant, vector-based designs through systematic tool usage.
You will be given a design plan that you must follow.

Core Workflow:
1. Create Canvas: Set up an appropriate canvas size using create_canvas()
2. Execute Design: Use add_cubic_bezier_curve() to build the design curve by curve
3. Modify Design: Use modify_cubic_bezier_curve() to update existing curves based on user feedback
4. List Curves: Use list_all_curves() to show current curves when user needs reference
5. Finalize: Call save_and_display_svg() to output the final design
6. Accept Human Feedback: Modify curves based on human feedback. Do not create a new canvas.
7. Finalize again, and accept more human feedback if necessary. This will happen in a loop until the design is complete.

Tool Usage Guidelines:

Canvas Creation:
- Follow the dimensions given in the design plan

Curve Implementation:
- Work systematically through the design plan
- Use descriptive variable names for curve parameters
- Consider these curve techniques:
  - Smooth curves: Control points roughly 1/3 distance from endpoints
  - Sharp corners: Place control points very close to endpoints
  - S-curves: Use opposing control point directions
  - Circles/arcs: Control points ~0.552 × radius from endpoints

Curve Modification:
- Use modify_cubic_bezier_curve() to update existing curves
- Always specify the correct curve_id (e.g., "curve_0", "curve_1")
- Use list_all_curves() to help users identify which curves to modify
- Preserve existing curve properties unless specifically asked to change them

Styling Best Practices:
- Use consistent stroke widths (typically 2-4px)
- Apply appropriate fill colors for closed shapes
- Consider using closed=True for complete shape outlines
- Use hex color codes for precise color matching

Design Principles:
- Minimalism: Use the fewest curves possible while maintaining quality
- Scalability: Design for vector scalability - avoid overly complex details
- Geometric Harmony: Align curves and maintain consistent proportions
- Professional Quality: Ensure clean, smooth curves without unnecessary complexity
"""

def create_agent():
    """Create the LangGraph agent for SVG logo design"""
    
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    llm_for_design = llm.with_structured_output(DesignPlan)
    
    def planning_node(state: AgentState) -> AgentState:
        """Node that plans the logo design"""
        logger.info("Planning logo design...")
        
        planning_prompt = f"""
        You are an expert SVG logo designer. Create a minimal logo design plan, that a downstream artist will later use to create the logo using cubic bezier curves.
        
        User request: {state.user_request}
        
        Your task:
        1. Plan a logo design that uses the MINIMAL number of cubic Bezier curves
        2. Each curve should have meaningful start/end points that connect logically. That means curves that connect should have connecting endpoints, or close to it.
        3. Each curve should have a unique id that we can use to reference it later.
        4. Consider the canvas size when positioning elements
        5. Think about visual balance and aesthetic appeal
        6. Write a text description of the design.
        """
        
        design_plan = llm_for_design.invoke([HumanMessage(content=planning_prompt)])
        
        return {"messages": [AIMessage(content=f"Design plan to follow: {design_plan.model_dump()}")]}
    

    def agentic_node(state: AgentState) -> AgentState:
        sys_message = SystemMessage(content=system_prompt)
        return {"messages": [llm_with_tools.invoke([sys_message] + state.messages)]}
    

    def summary_node(state: AgentState) -> AgentState:
        prompt = "Give the user a numbered list of the curves generated so far, giving the unique ID of the curve and a short description. Whatever is most helpful for users to reference and modify curves."
        raise NodeInterrupt(llm.invoke([prompt] + state.messages))
    

    def tools_or_human_feedback_condition(state) -> Literal["tools", "human_feedback"]:
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get("messages", [])):
            ai_message = messages[-1]
        elif messages := getattr(state, "messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "human_feedback"
    
    
    # Create the state graph
    memory = MemorySaver()
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("agent", agentic_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("human_feedback", summary_node)
    
    # Add edges
    workflow.add_edge(START, "planning")
    workflow.add_edge("planning", "agent")
    workflow.add_conditional_edges(
        "agent",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_or_human_feedback_condition,
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("human_feedback", "agent")
    
    return workflow.compile(checkpointer=memory)

def create_logo(user_request: str, filename: str = "logo_design.svg") -> str:
    """
    Main function to create a logo based on user request
    
    Args:
        user_request: Description of the desired logo
        filename: Output filename for the SVG
        
    Returns:
        SVG content as string
    """
    logger.info(f"Creating logo: {user_request}")
    
    # Create the agent
    agent = create_agent()
    
    # Initialize state
    initial_state = AgentState(
        user_request=user_request,
        filename=filename,
        messages=[HumanMessage(content=f"Create a logo: {user_request}")]
    )

    thread = {"configurable": {"thread_id": "5"}}

    # Run the graph until the first interruption
    for event in agent.stream(initial_state, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    
    # Get user input
    user_input = input("Tell me which curves you want to update: ")

    # We now update the state as if we are the human_feedback node
    agent.update_state(thread, {"messages": user_input}, as_node="human_feedback")

    for event in agent.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
    

# Example usage
if __name__ == "__main__":
    user_input = "man on the moon"
        
    if user_input:
        try:
            result = create_logo(user_input)
            print(f"\nDesign completed! Check the generated SVG file.")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Please enter a valid logo description.")