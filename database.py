import pymysql

def get_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='비밀번호',
        db='bookcalendar',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
