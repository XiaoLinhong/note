## 数据库

**数据库**是一堆有关联的数据集，如何描述关联？

## 数据模型
数据模型定义数据关系的范式。是数据结构的结构，比如

- 关系模型（Relational Model）
- 层次模型（Hierarchical Model）
- ...

## 关系模型
D 为域（集合），对于 一组域 \\( \\{ D_i | i = 1, 2, 3 ... \\} \\)，设
\\(a_i \in D_i\\)，称 \\( < a_1, a_i, ..., a_n > \\) 为定义在**这组域**上的元组。

> 这里的域，更准确率的数学描述应该是集合，一组域 => 集合的一个集合。 

**关系**是指 一组域 \\( \\{ D_i | i = 1, 2, 3 ... \\} \\)的笛卡尔积（组合中的笛卡尔积计数）构成的**全部元组集合** \\( \\{ < a_1, a_i, ..., a_n > | a_i \in D_i \\} \\) **中符合某种意义的元组子集合**。

> 关系是一种集合！

元组 \\( < a_1, a_i, ..., a_n > \\) 是关系 \\(R\\) 的成员。

关系 \\(R\\) 中所有元组某个维度构成的域 \\( A_i = \\{ a_i | a_i \in D_i \\} \\) ，为关系 \\(R\\)的 某个属性。

> 属性 \\(A_i\\) 是一种域（集合）！是域 \\(D_i\\) 的子域

**模式（关系模式）** 是关系的实例化后的实体，实例化是指给定每个 \\(A_i\\) 的具体定义以及关系名称。

定义在关系模型上最重要的操作是**关系代数**和**元组演算**。

## 关系代数

关系代数的运算 是定义在元组集合上的，是从集合到集合的映射（存在一元和二元运算）。

### 基本关系代数运算

基本关系代数运算是指无法通过其它运算推导出来的基本运算。

**1. 并（Union）**

关系代数中的并，就是集合的并运算。
 \\[ R \cup S \\]

条件：\\(R\\) 和 \\(S\\) 必须是**同构关系**（即属性个数和所属域一致）

**2. 差（Difference）**

差运算后的集合表示属于 $R$ 但不属于 $S$ 的元组集合。
 \\[R - S \\]

条件：\\(R\\) 和 \\(S\\) 必须是**同构关系**（即属性个数和所属域一致）

**3. 选择（Selection）**
 \\[ \sigma_{\theta}(R) \\]
通过条件 \\(\theta\\)，过滤集合 \\(R\\), 条件 \\(\theta\\) 为**元组属性的比较运算** + **布尔运算**，例如：

\\[\sigma_{price > 30}(Product)\\] 

**4. 投影（Projection）**
 \\[ \pi_{A_1, A_2, \dots, A_k}(R) \\]

 表示从关系 \\(R\\) 中选取属性 \\(A_1, A_2, \dots, A_k\\)。
 类似向量空间中的降维，从高纬度投影到低维度。

**5. 笛卡尔积（Cartesian Product）**
 
 \\[ R \times S \\]

笛卡尔积得到的是 \\(R\\) 与 \\(S\\) 元组的所有可能组合，结果元组数为 \\((r, s)\\)。

### 扩展代数关系运算

扩展代数运算可以有基本代数运算组合得到，定义扩张运算可以更加方便地写出表达式

**1. 交 （INTERSECT）**

关系代数中的交，就是集合的交运算

 \\[ R \cap S = R - ( R - S )\\]

条件：\\(R\\) 和 \\(S\\) 必须是**同构关系**（即属性个数和所属域一致）

**2. 连接（Join）**

θ-连接（Theta Join）
\\[R \bowtie_{\theta} S = \sigma_{\theta}(R \times S)\\]

相等连接
\\[R \bowtie_{e} S = \sigma_{e}(R \times S)\\]
相等连接是特殊的 θ-连接，条件 \\(\theta\\) 只能是等于表达式。

自然连接（Natural Join）
\\[R \bowtie S\\]
自动在相同属性名上进行匹配，连接后同名属性消失。

**3. 除法（Division）**

\\[R \div S = \\{ a \mid \forall b \in S, (a, b) \in R \\}\\]

表示那些与 \\(S\\) 中的所有 \\(b\\) 组合都在 \\(R\\) 中的 \\(a\\)。

\\[
R \div S = \pi_X(R) - \pi_X ( ( \pi_X(R) \times S ) - R)
\\]

举个例子，所有选修了 S 中所有课程的学生。

\\[R(Student, Course)\\]
\\[S(Course)\\]

\\[
   R \div S = \pi_{Student}(R) - \pi_{Student} ( ( \pi_{Student}(R) \times S ) - R)
\\]

## 元组演算
元组演算基于逻辑谓词，并使用元组变量来表达查询条件（集合的定义形式）。

查询结果是所有满足条件 \\( P(t) \\) 的 \\( t \\) 所组成的集合.
\\[\\{ t | P(t) \\}\\]

- \\( t \\) 是元组
- \\( P(t) \\) 是逻辑表达式，表示元组 \\( t \\) 应满足的条件

\\( P(t) \\) 的递归定义可以写作

``` BNF 
P(t) ::= 
    t.A θ c               // 属性与常量比较: 原子谓词
  | t1.A θ t2.B           // 两个元组的属性比较: 原子谓词
  | t ∈ R                // t 属于关系 R: 原子谓词
  | ¬P(t)                 // 逻辑非
  | P(t) ∧ Q(t)          // 逻辑与
  | P(t) ∨ Q(t)          // 逻辑或
  | ∃ s (P(s))           // 存在某个元组 s 满足
  | ∀ s (P(s))           // 所有元组 s 满足

  | P(t) → Q(t)           // 可选扩展项 ¬P(t) ∨ Q(t)
  | ∃ (s ∈ R) (P(s))     // 可选扩展项 ∃ s (s ∈ R ∧ P(s))
  | ∀ (s ∈ R) (P(s))     // 可选扩展项 ∀ s (s ∈ R → P(s))
```
其中
- \\( P(t) \\) 和 \\( Q(t) \\) 都是逻辑表达式。
- 元组 \\( t \\) 的属性 \\( A\\) 的值表示为 \\( t.A \\)
- \\( \exists \\) 为 存在（存在量词）
- \\( \forall \\) 为 所有（全称量词）
- 

举个例子，找出选修了所有 **张三** 任教课程的学生，
案例数据库的所有关系模式如下：
```
Student(sid, sname)
Course(cid, cname, teacher)
Enroll(sid, cid)
```

**采用最基础的语法书写**
\\[
\\{s | s \in Student \land
      \forall c (\lnot (c \in Course \land c.teacher = "张三") \lor
                (\exists e 
                     (e \in Enroll \land e.sid = s.sid \land e.cid = c.cid)
                   )
                )
\\}
\\]
对每门课程我们检查：
- 如果它不是张三的课，跳过（条件自然为真）
- 如果它是张三的课，必须存在一条选课记录，说明学生选了它

**加上蕴含扩展**
\\[
\\{s | s \in Student \land
      \forall c (c \in Course \land c.teacher = "张三" \rightarrow
                (\exists e 
                     (e \in Enroll \land e.sid = s.sid \land e.cid = c.cid)
                   )
                )
\\}
\\]

对每门课程我们检查：
- 如果它不是张三的课，跳过
- 如果它是张三的课，必须存在一条选课记录，说明学生选了它

**量词扩展定义域写法**。
\\[
\\{ s | s \in Student \land 
       \forall c \in \\{c | c \in Course \land c.teacher = "张三" \\}
              (\exists e \in Enroll 
                     (e.sid = s.sid \land e.cid = c.cid) )
\\}
\\]

对每门张三的课程我们检查：
- 必须存在一条选课记录，说明学生选了它

**用关系代数来写查询**
\\[
   Student \Join
   (\pi_{sid, cid}(Student \Join Enroll) \div
   \pi_{cid}(\sigma_{teacher="张三"}(Course)))
\\]

## 约束条件
候选码：每个元组都不一样最小属性集（唯一性，最小性）

可以选取一个候选码作为主键（Primary Key）。
外键是一种关系的特殊属性，该属性是其他关系的主键。

- 约束条件1: 主键属性不能为未定义
- 约束条件2: 外键的属性必须满足外键的已取值范围（不是定义域的范围）
- 约束条件3: 属性定义域的约束条件

## 数据库管理系统
数据库管理系统一种软件实现（比如 mysql），
用来管理 多个 基于某种 **数据模型**的数据库的软件。

## 关系数据库管理系统

基于关系模型的 数据库管理系统实现（比如 Postgres ）。

在用户层面希望看到的功能。

- **用户信息**的增删改查（权限管理）
- **数据库**的增删改查（多张表，依靠外键关联？）
- **关系/表**的增删改查（ schema/模式 ）
- **数据**的增删改查（view/data: 行/元组）

增删改查的ACID特性：原子性、一致性、隔离性、持久性

在关系数据库管理系统实现的关系数据结构叫做**表**。
表可以看作是关系的一种表达形式（从集合 到 规则的长方形表格）。

### 索引

索引是关系数据库中对某一列或多个列的值进行预排序的数据结构，
为了加速查询操作而创建的数据结构。

主键是一种特殊的索引。

### 视图

视图（View）是关系型数据库中的一种虚拟表，它本身不存储数据，而是对一个或多个真实表（或其他视图）的查询结果进行封装。 

可以屏蔽掉 **真实表** 的细节（为用户提供一致的表结构，真实表改了，视图可以不变）。
