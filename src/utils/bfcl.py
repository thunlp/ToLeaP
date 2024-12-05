
import re
from ast import literal_eval

def parse_parameter_value(value_string):
    value_string = value_string.strip()
    if value_string.startswith('"') and value_string.endswith('"'):
        return value_string[1:-1]
    elif value_string.startswith('\'') and value_string.endswith('\''):
        return value_string[1:-1]
    elif value_string.isdigit():
        return float(value_string)
    elif value_string.startswith('[') or value_string.startswith('{'):
        return literal_eval(value_string)
    else:
        return value_string.lower() == "true"

def process_multiple_func_string(func_string):
    def no_name(func_string):
        return '=' not in func_string and '()' not in func_string
    is_no_name = no_name(func_string)
    func_name = func_string.split('(')[0].strip()
    parameter_string = re.search(r"\((.*?)\)", func_string).group(1)
    parameters = {}
    if is_no_name:
        parameters[""] = parameter_string
    else:
        current_param = ""
        current_value = ""
        in_quotes = False
        quote_char = None
        in_brackets = 0
        in_value = False
        
        for char in parameter_string + ',':  # Add comma to handle last parameter
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                current_value += char
            elif char in '[{':
                in_brackets += 1
                current_value += char
            elif char in ']}':
                in_brackets -= 1
                current_value += char
            elif char == '=' and not in_quotes and in_brackets == 0:
                in_value = True
                current_param = current_param.strip()
            elif char == ',' and not in_quotes and in_brackets == 0:
                if in_value:
                    current_value = current_value.strip()
                    parameters[current_param] = parse_parameter_value(current_value)
                current_param = ""
                current_value = ""
                in_value = False
            else:
                if in_value:
                    current_value += char
                else:
                    current_param += char
    
    return {"name": func_name, "arguments": parameters}