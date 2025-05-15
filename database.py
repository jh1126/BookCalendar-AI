import pymysql
import os

def load_password(path):
    with open(path, "r") as file:
        token = file.read().strip()
    return password
    
def get_connection():
    
    pass_path = os.path.join(os.path.dirname(__file__), 'secret', 'db_pass.txt')
    pass_path = os.path.abspath(pass_path)
    
    return pymysql.connect(
        host='localhost',
        user='bookcalendar123',
        password=load_password(pass_path),
        db='bookcalendar',
        port=3307,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
