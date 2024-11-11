import json

def format_sharegpt_tools(json_str):
    """
    Convert JSON tool definitions to human-readable documentation format.
    
    Args:
        json_str (str): JSON string containing tool definitions
        
    Returns:
        str: Formatted documentation string
    """
    # Parse JSON string
    tools = json.loads(json_str)
    
    # Store formatted strings
    formatted_docs = []
    
    for tool in tools:
        # Start with function name and description
        doc = f"{tool['name']}(params): {tool['description']}"
        
        # Get parameters
        params = tool['parameters'].get('properties', {})
        
        # Only add Parameters section if there are parameters
        if params:
            doc += "\nParameters:"
            for param_name, param_info in params.items():
                param_desc = param_info.get('description', 'No description provided')
                doc += f"\n{param_name} - {param_desc}"
        
        formatted_docs.append(doc)
    
    # Join all documentation blocks with double newlines
    return "\n\n".join(formatted_docs)