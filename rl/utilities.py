import re

def parse_latex_to_python(expr):
    # Remove commas from large numbers
    expr = expr.replace(',', '')
    
    # Handle LaTeX fractions (\frac{num}{denom}) => num/denom
    expr = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'(\1)/(\2)', expr)
    
    # Handle square roots (\sqrt{100}) => math.sqrt(100)
    expr = re.sub(r'\\sqrt{([^{}]+)}', r'math.sqrt(\1)', expr)
    
    # Handle nth roots (\sqrt[n]{100}) => math.pow(100, 1/n)
    expr = re.sub(r'\\sqrt\[(\d+)\]{([^{}]+)}', r'math.pow(\2, 1/\1)', expr)
    
    # Handle exponents a^b => a**b
    expr = re.sub(r'([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)', r'\1**\2', expr)
    
    # Handle trigonometric functions (e.g., \sin{x} => math.sin(x))
    expr = re.sub(r'\\sin{([^{}]+)}', r'math.sin(\1)', expr)
    expr = re.sub(r'\\cos{([^{}]+)}', r'math.cos(\1)', expr)
    expr = re.sub(r'\\tan{([^{}]+)}', r'math.tan(\1)', expr)
    
    # Handle logarithmic functions (\log{a}{b} => math.log(b, a))
    expr = re.sub(r'\\log\{([^{}]+)\}\{([^{}]+)\}', r'math.log(\2, \1)', expr)
    
    # Handle exponential function (\exp{x} => math.exp(x))
    expr = re.sub(r'\\exp{([^{}]+)}', r'math.exp(\1)', expr)
    
    # Handle natural logarithm (\ln{x} => math.log(x))
    expr = re.sub(r'\\ln{([^{}]+)}', r'math.log(\1)', expr)
    
    # Handle text expressions (\text{Hello} => Hello)
    expr = re.sub(r'\\text{([^{}]+)}', r'\1', expr)
    
    try:
        return float(eval(expr))
    except Exception:
        return expr.strip()  # Return stripped string if not a number

def compare_latex_numbers(expr1, expr2):
    parsed1 = parse_latex_to_python(expr1)
    parsed2 = parse_latex_to_python(expr2)
    
    if isinstance(parsed1, float) and isinstance(parsed2, float):
        return abs(parsed1 - parsed2) < 1e-9  # Compare numeric values
    else:
        return parsed1 == parsed2  # Compare as strings
    
if __name__ == '__main__':
    a = r'\frac{1}{2}'
    b = r'0.5'
    print(compare_latex_numbers(a, b))  # True
    a = r'\sqrt{100}'
    b = r'10'
    print(compare_latex_numbers(a, b))  # True
    a = r'\sqrt[3]{1,000}'
    b = r'100'
    print(compare_latex_numbers(a, b))  # False
    a = r'\text{Hello}'
    b = r'Hello'
    print(compare_latex_numbers(a, b))  # True
