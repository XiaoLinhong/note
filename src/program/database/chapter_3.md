# postgresql

## 安装
下载
``` bash
wget https://ftp.postgresql.org/pub/source/v17.4/postgresql-17.4.tar.gz
```

## 启动
初始化一个目录为数据库目录
``` bash
initdb -D /public/home/bedrock/envs/database
```

开始数据库服务
``` bash
pg_ctl -D /public/home/bedrock/envs/database -l logfile start
```

``` bash
[bedrock@node1 database]$ netstat -nplt  | grep postgres
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
tcp        0      0 127.0.0.1:5432          0.0.0.0:*               LISTEN      14276/postgres      
tcp6       0      0 ::1:5432                :::*                    LISTEN      14276/postgres 
```

命令行连接数据库
```
psql -U bedrock -h localhost -d postgres
```

远程连接
``` bash 
host    all all 0.0.0.0/0   md5 # pg_hba.conf
host    all all ::/0        md5 # pg_hba.conf

listen_addresses = '*' # postgresql.conf
```

```
pg_ctl -D /public/home/bedrock/envs/database -l logfile restart
```

## 用法

``` python
import psycopg2

serverKargs = dict( # 连接参数
    host = 'localhost',
    port = '5432',
    user = 'xiaolh',
    password = 'xiaolh123456',
    database = 'obs',
)

with psycopg2.connect(**serverKargs) as fh:
    with fh.cursor() as cursor:    
        cursor.execute("SELECT version();")
        print("PostgreSQL info:", cursor.fetchone())
        
        # 插入
        data = [("demo1", "demo1@expm.com"), ("demo2", "demo2@expm.com"), ("demo3", "demo3@expm.com")]
        cursor.executemany("INSERT INTO users (name, email) VALUES (%s, %s)", data)
        fh.commit()  # 提交更改

        # 删除
        cursor.execute("DELETE FROM users WHERE name = %s", ("demo3",))
        fh.commit()  # 提交更改

        # 更新
        oldName, newName = "demo1", "xiaolh1"
        cursor.execute("UPDATE users SET name = %s WHERE name = %s", (newName, oldName))
        fh.commit()  # 提交更改

        # 查看
        cursor.execute("SELECT * FROM users LIMIT 15")
        for row in cursor.fetchall():
            print(row)
```
