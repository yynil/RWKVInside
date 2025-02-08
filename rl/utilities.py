import re
import math

def parse_coordinate(coord_str):
    """解析坐标字符串，返回(x, y)元组"""
    # 移除所有空白字符
    cleaned = re.sub(r'\s+', '', coord_str)
    
    # 匹配坐标格式 (x,y)
    match = re.match(r'\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)', cleaned)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    return None

def parse_latex_to_python(expr):
    if isinstance(expr, (int, float)):
        return float(expr)
        
    # Remove unnecessary spaces but keep commas in coordinates
    expr = re.sub(r'\s+', '', expr)
    
    # 优先尝试解析坐标
    coord = parse_coordinate(expr)
    if coord is not None:
        return coord
        
    # 现在移除逗号（仅对非坐标的情况）
    expr = expr.replace(',', '')
    
    # Handle LaTeX fractions (\frac{num}{denom}) => num/denom
    expr = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'(\1)/(\2)', expr)
    
    # Handle LaTeX fractions (\dfrac{num}{denom}) => num/denom
    expr = re.sub(r'\\dfrac{([^{}]+)}{([^{}]+)}', r'(\1)/(\2)', expr)
    
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
    
    #Handle currency expressions ($,£, €, ¥,฿,20.00 => 20.00, ,etc.)
    expr = re.sub(r'\$([0-9.]+)', r'\1', expr)
    expr = re.sub(r'£([0-9.]+)', r'\1', expr)
    expr = re.sub(r'€([0-9.]+)', r'\1', expr)
    expr = re.sub(r'¥([0-9.]+)', r'\1', expr)
    expr = re.sub(r'฿([0-9.]+)', r'\1', expr)
    
    
    
    try:
        result = eval(expr)
        return float(result) if isinstance(result, (int, float)) else result
    except Exception:
        return expr.strip()

def compare_latex_numbers(expr1, expr2):
    parsed1 = parse_latex_to_python(expr1)
    parsed2 = parse_latex_to_python(expr2)
    
    # 处理坐标比较
    if isinstance(parsed1, tuple) and isinstance(parsed2, tuple):
        return (abs(parsed1[0] - parsed2[0]) < 1e-6 and 
                abs(parsed1[1] - parsed2[1]) < 1e-6)
    
    # 处理数值比较
    if isinstance(parsed1, (int, float)) and isinstance(parsed2, (int, float)):
        return abs(parsed1 - parsed2) < 1e-6
        
    # 处理字符串比较
    return str(parsed1) == str(parsed2)
    
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
    # 测试坐标比较
    print(compare_latex_numbers('(0, 10)', '(0,10)'))  # True
    print(compare_latex_numbers('(0.0, 10.0)', '(0,10)'))  # True
    print(compare_latex_numbers('(0,      10)', '(0,10)'))  # True
    print(compare_latex_numbers('(1, 10)', '(0,10)'))  # False

    a = r'$20.00'
    b = '20'
    print(compare_latex_numbers(a, b))  # True
    
    a = r'\dfrac{1}{2}'
    b = r'0.5'
    print(compare_latex_numbers(a, b))  # True
    
    a = r'\dfrac{1}{2}'
    b = r'\frac{1}{2}'
    print(compare_latex_numbers(a, b))  # True