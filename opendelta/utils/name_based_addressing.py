from typing import List
import re
def superstring_in(str_a: str , list_b: List[str]):
    r"""check whether there is any string in list b containing str_a.

    Args:
    Returns:
    """
    return any(str_a in str_b for str_b in list_b)

def is_child_key(str_a: str , list_b: List[str]):
    r"""check whether a string in ``list_b`` is the child key in ``str_a``

    Args:
    Returns:
    """
    return any(str_b in str_a and (str_b==str_a or str_a[len(str_b)]==".") for str_b in list_b)

def endswith_in(str_a: str , list_b: List[str]):
    r"""check whether ``str_a`` has a substring that is in list_b.

    Args:
    Returns:
    """
    return any(str_a.endswith(str_b) and (str_a==str_b or str_a[-len(str_b)-1] == ".")  for str_b in list_b)

def substring_in(str_a: str , list_b: List[str]):
    r"""check whether ``str_a`` has a substring that is in list_b.

    Args:
    Returns:
    """
    token_a = str_a.split(".")
    for str_b in list_b:
        token_b = str_b.split(".")
        for i in range(len(token_a)-len(token_b)+1):
            if "".join(token_a[i:i+len(token_b)]) == "".join(token_b):
                return True
    return False

def endswith_in_regex(str_a: str , list_b: List[str]):
    r"""check whether ``str_a`` has a substring that is in list_b.

    Args:
    Returns:
    """
    for str_b in list_b:
        ret = re.search(str_b, str_a)
        if ret is not None:
            b = ret.group()
            if ret.span()[1] == len(str_a) and (b == str_a or (str_a==b or str_a[-len(b)-1] == ".")):  
                # the latter is to judge whether it is a full sub key in the str_a, e.g. str_a=`attn.c_attn` and list_b=[`attn`] will given False
                return True
    return False

def substring_in_regex(str_a: str , list_b: List[str]):
    r"""check whether ``str_a`` has a substring that is in list_b.

    Args:
    Returns:
    """
    for str_b in list_b:
        ret = re.search(str_b, str_a)
        if ret is not None:
            b = ret.group()
            if (ret.span()[0] == 0 or str_a[ret.span()[0]-1] == ".") and \
                (ret.span()[1] == len(str_a) or str_a[ret.span()[1]] == "."): #and b == str_a and (str_a==b or str_a[-len(b)-1] == "."):  
                # the latter is to judge whether it is a full sub key in the str_a, e.g. str_a=`attn.c_attn` and list_b=[`attn`] will given False
                return True
    return False
    