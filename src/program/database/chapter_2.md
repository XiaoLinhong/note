# SQL
SQL（Structured Query Language，结构化查询语言）是一种用于管理关系型数据库的标准语言。

SQL 是一种声明式语言，是构建在元组演算和关系代数之上的。

SQL 可以分为 **数据操作语言（DML, Data Manipulation Language）**、
**数据定义语言（DDL, Data Definition Language）** 和 **数据控制语言（DCL, Data Control Language）**

## 数据控制语言

### 用户的增删改查

**增加**
``` sql
CREATE USER username WITH PASSWORD 'password';
CREATE ROLE xiaolh WITH LOGIN PASSWORD 'xiaolh123456'; -- 或更常用：
```

**删除**
``` sql
DROP USER username;
DROP ROLE username;
```

**修改**
``` sql
ALTER USER username WITH PASSWORD 'newpassword';
ALTER USER username RENAME TO new_username;
ALTER USER username WITH SUPERUSER;         -- 赋予超级权限
ALTER USER username WITH NOSUPERUSER;       -- 撤销超级权限
```

权限管理
``` sql
-- 增加
GRANT SELECT, INSERT ON tbname TO username;
GRANT ALL PRIVILEGES ON DATABASE dbname TO username;

ALTER USER xiaolh WITH CREATEDB; -- 创建库的权限
ALTER USER xiaolh WITH NOCREATEDB;

-- 回收
REVOKE SELECT ON tbname FROM username;
REVOKE ALL PRIVILEGES ON DATABASE dbname FROM username; -- 给表的所有权限
```
常见权限有``SELECT	``, `` INSERT ``, `` UPDATE ``, `` DELETE ``, ``ALL PRIVILEGES `。

**查看**
``` sql
\du  -- 查看用户及权限
SELECT * FROM pg_roles;
```

## 数据定义语言

### 数据库的增删改查

**增加**
``` sql
CREATE DATABASE dbname;
```

**删除**
``` sql
DROP DATABASE dbname;
```

**修改**
``` sql
ALTER DATABASE dbname RENAME TO new_dbname;
ALTER DATABASE dbname OWNER TO new_owner; -- 给其他用户
```

**查看**
``` sql
\l            -- 查看所有数据库（在 psql 中）
SELECT datname FROM pg_database; -- SQL 查询数据库名
```

### 数据表的增删改查

**增加**
``` sql
CREATE TABLE tbname (
    列名1 数据类型 [约束],
    列名2 数据类型 [约束],
    ...
);
```
常见约束有``PRIMARY KEY	``, `` UNIQUE ``, `` NOT NULL ``, `` CHECK ``, ``FOREIGN KEY``,

~~~ admonish example
``` sql
CREATE TABLE score (
    student_id INT,
    course_id INT,
    score FLOAT CHECK (score >= 0 AND score <= 100),
    FOREIGN KEY (student_id) REFERENCES student(id)
);
```
~~~

**删除**
``` sql
DROP TABLE tbname;
TRUNCATE TABLE tbname; -- 清空表中所有数据，但保留表结构
```

**修改**
``` sql
ALTER TABLE tbname RENAME TO new_name; -- 修改表格名称
ALTER TABLE tbname ADD COLUMN age INT; -- 添加字段
ALTER TABLE tbname DROP COLUMN age; -- 删除字段
ALTER TABLE tbname RENAME COLUMN username TO new_name;  -- 修改字段名称
ALTER TABLE tbname ALTER COLUMN username TYPE VARCHAR(100); -- 修改字段类型
```

索引
``` sql
CREATE INDEX idx_name ON tbname(username);
DROP INDEX idx_name;
```

主键/外键
``` sql
ALTER TABLE tbname ADD CONSTRAINT 约束名 PRIMARY KEY(id);
ALTER TABLE tbname ADD CONSTRAINT 约束名 FOREIGN KEY(office_id) REFERENCES office(id);
```

**查看**
``` sql
\di                -- psql 命令 查询索引
\dt                -- psql 命令 查看当前数据库所有表
\d tbname          -- psql 命令 查看表结构
\z tbname          -- psql 命令 查看表权限
```

## 数据操作语言
主要命令：

- INSERT：向表中插入数据。
- DELETE：删除表中的数据。
- UPDATE：更新表中的数据。
- SELECT：从一个或多个表中检索数据。

### 向表中插入数据

语法
``` sql
INSERT INTO table_name (column1, column2) VALUES (value1, value2);
```

~~~ admonish example
``` sql
INSERT INTO students (class_id, name, gender, score) VALUES
(2, 'name1', 'M', 80),
(2, 'name2', 'M', 81);
```
~~~

### 删除表中的数据

语法
``` sql
DELETE FROM table_name WHERE condition;
```

``` sql example
DELETE FROM students WHERE id=1;
```

### 更新表中的数据

语法
``` sql
UPDATE table_name SET column1 = value1 WHERE condition;
```

~~~ admonish  example
``` sql
UPDATE students SET score=score+10 WHERE score<80;
```
~~~

### 检索库中的数据
语法
``` sql
SELECT [DISTINCT] 列表达式
FROM 表或子查询
[JOIN 子句]
[WHERE 条件]
[GROUP BY 分组列 [HAVING 条件]]
[ORDER BY 排序列 [ASC|DESC]]
[LIMIT 限制行数 [OFFSET起始行]]
```

**注意事项**
- 字符的 WHERE 条件，可以使用通配符进行模糊查询，LIKE（模糊查询） 和 _，%（通配符）
- JOIN 子句，分为 [INNER/LEFT/RIGHT/FULL] JOIN（自连接，内连接，外连接：处理null）
- HAVING 条件 是针对 GROUP BY 后的集合
- 列表达式是指 元组属性的算数表达式 + 聚合函数（MIN、MAX、AVG、SUM等），元组属性本身也是列表达式
- AS 赋值语句，可以把表达式的值进行赋值，或者重命名属性/表名称
- FROM 中的子查询 可以看作一种临时的视图（本应该为表名的地方，变为一个动态的匿名表）
- WHERE 中子查询有三种，包括 IN, \\(\theta - all/some\\) 和 EXISTS
- 基于集合操作，UNION, INTERSECT, EXCEPT

其他扩展语法
- CASE WHEN 逻辑判断（等价于 if-else）

**简单案例**
~~~ admonish example
``` sql
SELECT Title FROM movies 
where Director = "John Lasseter" 
order by year Desc 
limit 1 offset 3
```
~~~

**一些简单的子查询案例**
~~~ admonish example
选出所在院系在北京的学生。
``` sql
SELECT name
FROM Student
WHERE deptID IN (
    SELECT deptID
    FROM Department
    WHERE location = 'Beijing'
);
```
~~~

元组演算表示：
\\[
\\{
    s.name \mid s \in Student \land \exists \  deptID \in 
    \\{ d.deptID \mid d \in Department \land d.location = 'Beijing' \\}
    \ (s.deptID = deptID)
\\}
\\]


~~~ admonish example
选出比 3 号院系所有学生分数都高的学生。
``` sql
SELECT name
FROM Student
WHERE score > ALL (
    SELECT score
    FROM Student
    WHERE deptID = 3
);
```
~~~

元组演算表示：

\\[
\\{ 
    s.name \mid s \in Student \land \forall score \in 
    \\{ t.score \mid t \in Student \land t.deptID = 3 \\} \ 
    (s.core > score)
\\}
\\]

注意 **\\(\theta - all/some \\) 子句** 与 元组演算的**定义域扩展语法** 有着很好的对于关系。

~~~ admonish example
选出分数小于 3 号院系某些学生的学生。
``` sql
SELECT name
FROM Student
WHERE score < SOME (
    SELECT score
    FROM Student
    WHERE deptID = 3
);
```
~~~

元组演算表示：
\\[
\\{
    s.name \mid s \in Student \land \exists \  score \in 
    \\{ t.score \mid t \in Student \land t.deptID == 3 \\}
    \ (s.score < score)
\\}
\\]

**关于关系代数的案例**

~~~ admonish example
交
``` sql
SELECT name FROM Student
INTERSECT
SELECT name FROM Course;
```
并
```
SELECT name FROM Student
UNION
SELECT name FROM Course;
```
~~~
