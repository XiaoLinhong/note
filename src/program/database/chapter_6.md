# 数据建模

在实际开发中，设计数据库是有一些规律和方法可循的，其中比较常见、也比较有名的一种方法就是使用 ER 模型（实体-关系模型）。

从需求描述到最终写出 SQL 代码，其实就是一个不断消除模糊、逐步明确细节的过程。而 ER 模型提供了一种清晰的思考框架，它引导我们从**实体**和**关系**这两个角度出发，去理清需求中涉及的对象和它们之间的联系，从而让原本不清楚的地方变得明确起来。

实体（Entity）表示一个独立、具体的对象。我们可以粗略地把实体理解为名词，比如“计算机”、“雇员”、“歌曲”或“数学定理”。

关系（Relationship）用来描述两个或多个实体之间是如何关联的。可以粗略地把关系看作是动词，比如“拥有”、“雇佣”、“演唱”或“引用”。

如何确定**需要哪些实体**、**哪些是关系**，其实是一门艺术。选择恰当的实体和关系，不仅能帮助我们更深入地理解需求，还能设计出结构更清晰、效率更高的数据库表。

这么说很抽象，我们以一个假想的项目，来走通从需求到观念模型，再到数据模型。

## 视频知识付费平台 (MindTrail)

需要满足的需求
- 平台支持用户的注册和管理。
- 不同用户有不同的角色（学生、老师、管理员、审核员）和权限。
- 教师发布课程视频，学生购买课程（打包视频），也可以购买某个视频，课程视频不能高于所有时评的价格总和。
- 一个课程有多个视频，学生购买后，需要记录学习进度。
- 审核员审核教师发布的视频。
- 管理员有权对用户、课程、视频进行管理（增删改查）。

对象有
- 角色，属性有: 角色ID，名称，权限级别。
- 用户，属性有: 用户ID，名称，性别，自我介绍，邮件，电话，出生年月，角色ID，创建时间
- 学科，属性有: 学科ID，名称，学科介绍
- 课程，属性有: 课程ID，名称，学科ID，课程简介，价格，状态（新建、发布、删除），创建时间
- 视频，属性有: 视频ID，名称，学科ID，视频简介，章节，状态（完成上传、提交审核、完成审核、完成发布、删除），价格，课程ID，上传时间
- 产品，属性有：产品ID，产品类型（视频、或者课程），课程ID，视频ID，价格

关系
- 视频归档到课程，老师可以把视频打包到某个课程（多对多的关系）。属性：归档ID，视频ID，课程ID
- 学生购买产品ID，属性：账单ID，用户ID，产品ID，价格，状态（创建，完成付款），付款时间。
- 审核员审核视频，属性：审核单ID，用户ID，视频ID，状态（待审核，完成审核）
- 学生学习课程，属性：学习ID，用户ID，产品ID，状态（未开始，学习中，完成），进度（第几个视频的什么位置）

E-R图如下

``` mermaid
erDiagram
    roles {
        int role_id
        varchar name
        int permission_level
    }

    users {
        int user_id
        varchar name
        enum gender
        text bio
        varchar email
        varchar phone
        date birthdate
        int role_id
        datetime created_at
    }

    subjects {
        int subject_id
        varchar name
        text description
    }

    courses {
        int course_id
        varchar name
        int subject_id
        text description
        decimal price
        enum status
        datetime created_at
    }

    videos {
        int video_id
        varchar name
        int subject_id
        text description
        varchar chapter
        enum status
        decimal price
        int teacher_id
        datetime uploaded_at
    }

    video_course_archive {
        int archive_id
        int video_id
        int course_id
    }

    products {
        int product_id
        enum product_type
        int video_id
        int course_id
        decimal price
    }

    purchases {
        int purchase_id
        int user_id
        int product_id
        decimal price
        enum status
        datetime paid_at
    }

    audits {
        int audit_id
        int user_id
        int video_id
        enum status
        datetime audited_at
    }

    study_progress {
        int progress_id
        int user_id
        int product_id
        enum status
        varchar progress_detail
        datetime updated_at
    }

    %% 关系
    users }o--|| roles : "has role"
    users ||--o{ purchases : "makes"
    users ||--o{ audits : "reviews"
    users ||--o{ study_progress : "studies"
    users ||--o{ videos : "uploads"
    
    subjects ||--o{ courses : "includes"
    subjects ||--o{ videos : "includes"

    videos ||--o{ video_course_archive : "linked to"
    courses ||--o{ video_course_archive : "linked to"

    videos ||--o{ products : "can be product"
    courses ||--o{ products : "can be product"

    products ||--o{ purchases : "is purchased"
    products ||--o{ study_progress : "is studied"

    videos ||--o{ audits : "are reviewed"
```

创建表格和数据

``` sql
-- 建立数据库
-- CREATE DATABASE mindtrail;

-- =============== 枚举类型定义 ===============

-- 性别类型
CREATE TYPE gender_enum AS ENUM ('男', '女');

-- 视频状态
CREATE TYPE video_status_enum AS ENUM (
    '完成上传', '提交审核', '完成审核', '完成发布', '删除'
);

-- 课程状态
CREATE TYPE course_status_enum AS ENUM (
    '新建', '发布', '删除'
);

-- 产品类型
CREATE TYPE product_type_enum AS ENUM (
    '视频', '课程'
);

-- 购买状态
CREATE TYPE purchase_status_enum AS ENUM (
    '创建订单', '完成付款'
);

-- 审核状态
CREATE TYPE audit_status_enum AS ENUM (
    '待审核', '完成审核'
);

-- 学习状态
CREATE TYPE study_status_enum AS ENUM (
    '未开始', '学习中', '完成'
);

-- =============== 主体表 ===============

-- 角色
CREATE TABLE roles (
    role_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    permission_level INT NOT NULL
);

-- 用户
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    gender gender_enum,
    bio TEXT,
    email VARCHAR(100),
    phone VARCHAR(20),
    birthdate DATE,
    role_id INT REFERENCES roles(role_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 学科
CREATE TABLE subjects (
    subject_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT
);

-- 课程
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    subject_id INT REFERENCES subjects(subject_id),
    description TEXT,
    price NUMERIC(10,2) NOT NULL DEFAULT 0,
    status course_status_enum DEFAULT '新建',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 视频
CREATE TABLE videos (
    video_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    subject_id INT REFERENCES subjects(subject_id),
    description TEXT,
    chapter VARCHAR(100),
    status video_status_enum DEFAULT '完成上传',
    price NUMERIC(10,2) NOT NULL DEFAULT 0,
    course_id INT,  -- 可选冗余字段，如果需要快速查询
    teacher_id INT REFERENCES users(user_id),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 产品（课程或视频）
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_type product_type_enum NOT NULL,
    course_id INT REFERENCES courses(course_id),
    video_id INT REFERENCES videos(video_id),
    price NUMERIC(10,2) NOT NULL
);

-- =============== 关系表 ===============

-- 视频归档到课程（多对多）
CREATE TABLE video_course_archive (
    archive_id SERIAL PRIMARY KEY,
    video_id INT NOT NULL REFERENCES videos(video_id),
    course_id INT NOT NULL REFERENCES courses(course_id)
);

-- 购买记录（产品）
CREATE TABLE purchases (
    purchase_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    product_id INT NOT NULL REFERENCES products(product_id),
    price NUMERIC(10,2) NOT NULL,
    status purchase_status_enum DEFAULT '创建',
    paid_at TIMESTAMP
);

-- 视频审核
CREATE TABLE audits (
    audit_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id), -- 审核员
    video_id INT NOT NULL REFERENCES videos(video_id),
    status audit_status_enum DEFAULT '待审核',
    audited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 学习进度
CREATE TABLE study_progress (
    progress_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id),
    product_id INT NOT NULL REFERENCES products(product_id),
    status study_status_enum DEFAULT '未开始',
    progress_detail VARCHAR(200),  -- 示例：视频3-00:14:52
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

```

插入一些假数据
``` sql
INSERT INTO roles (name, permission_level) VALUES
('学生',     1),
('老师',     2),
('审核员',   3),
('管理员',   4);

INSERT INTO users (name, gender, bio, email, phone, birthdate, role_id) VALUES
('张三', '男', '热爱学习', 'zhangsan@example.com', '13800000001', '2000-01-01', 1),
('李四', '女', '资深教师', 'lisi@example.com', '13800000002', '1990-02-15', 2),
('王五', '男', '审核员', 'wangwu@example.com', '13800000003', '1995-03-25', 3),
('赵六', '女', '管理员', 'zhaoliu@example.com', '13800000004', '1985-04-30', 4);

INSERT INTO subjects (name, description) VALUES
('数学', '关于数学的课程'),
('物理', '关于物理的课程'),
('编程', '学习编程的课程');

INSERT INTO courses (name, subject_id, description, price, status) VALUES
('高等数学', 1, '学习高等数学的基础课程', 199.99, '发布'),
('大学物理', 2, '物理学的基础课程', 159.99, '发布'),
('Python 编程入门', 3, 'Python 编程的入门课程', 129.99, '新建');

UPDATE courses SET user_id = 2;

INSERT INTO videos (name, subject_id, description, chapter, status, price, course_id, teacher_id) VALUES
('高等数学 - 第一章', 1, '高等数学第一章的基础讲解', '第一章', '完成上传', 49.99, 1, 2),
('高等数学 - 第二章', 1, '高等数学第二章讲解', '第二章', '提交审核', 49.99, 1, 2),
('大学物理 - 第一章', 2, '物理学第一章的基础讲解', '第一章', '完成审核', 59.99, 2, 2),
('Python 编程入门 - 第1节', 3, 'Python 编程的基础知识', '第1节', '完成上传', 39.99, 3, 2);

INSERT INTO products (product_type, course_id, video_id, price) VALUES
('课程', 1, NULL, 199.99),
('课程', 2, NULL, 159.99),
('视频', NULL, 1, 49.99),
('视频', NULL, 2, 49.99),
('视频', NULL, 3, 59.99),
('视频', NULL, 4, 39.99); 

INSERT INTO purchases (user_id, product_id, price, status, paid_at) VALUES
(1, 1, 199.99, '完成付款', '2025-04-01 10:00:00'),
(2, 2, 159.99, '完成付款', '2025-04-02 11:00:00'),
(3, 3, 49.99, '创建订单', NULL),
(4, 4, 39.99, '完成付款', '2025-04-03 12:00:00');

INSERT INTO audits (user_id, video_id, status, audited_at) VALUES
(3, 2, '完成审核', '2025-04-02 14:00:00'),
(3, 3, '完成审核', '2025-04-03 15:00:00');


INSERT INTO study_progress (user_id, product_id, status, progress_detail, updated_at) VALUES
(1, 1, '学习中', '视频1-00:05:00', '2025-04-01 12:00:00'),
(2, 2, '未开始', NULL, '2025-04-02 13:00:00'),
(4, 4, '已完成', '视频4-01:30:00', '2025-04-03 13:00:00');
```

进行查询，所有学生及其购买的课程。
``` sql
SELECT u.user_id, u.name AS student_name, c.name AS course_name, p.price AS course_price, pu.status AS purchase_status
FROM users u
JOIN purchases pu ON u.user_id = pu.user_id
JOIN products p ON pu.product_id = p.product_id
JOIN courses c ON p.course_id = c.course_id
WHERE u.role_id = 1;
```

