
def f(path: str, t1=0, t2=1):
    return path[t1:t2]


dic = {"path": "coucou", "t1": 2, "t2": 3}

print(f(**dic))  # cou
print(dic["cd"])
