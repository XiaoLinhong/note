## mdbook 的备忘录

### 1. 命令

 - 开始：``` mdbook init [path/to/book] [--title "some"] [--ignore] ```
 - 渲染：``` mdbook build [-o] ```，默认当前工作目录就是根目录
 - 调试：``` mdbook watch [-o] ```，默认当前工作目录就是根目录
 - 调试：``` mdbook clean ```，默认当前工作目录就是根目录
 - 服务：``` mdbook serve path/to/book -p 8000 -n 127.0.0.1 ```

### 2. 结构

``` SUMMARY.md ```用来表明层次结构，并导入章节对应的 Markdown 文件。
其格式非常严格，具体结构如下：

``` markdown 
# Summary <!-- 标题, 可选 -->

--- <!-- 生成 HTML 渲染行 -->

[前言](./prefix.md)  <!-- 前缀章节, 可选 -->


<!-- 可以没有标题[]() -->
- [文档工具]() <!-- 草稿章节, HTML 中的禁用链接 -->
    - [mdbook 的备忘录](./docs/chapter_1.md) <!-- 编号章节, 可以嵌套 -->

[后记](./Suffix.md) <!-- 后缀章节, 可选 -->
``` 

注意 章节对应的 Markdown 文件不存在 ``` mdbook ``` 会自动创建空文件

### 3. 配置
``` book.toml ``` 为整本书的配置文件

``` toml
[book]
authors = ["xiaolh"]
language = "cn"
multilingual = false # 是否支持多语言
src = "src"
title = "xiao's notes"
description = "mdbook" # 作为元信息添加到 html<head>中

# 默认预处理
[preprocessor.index] # 将所有名为 README.md 的章节文件转换为 index.md
[preprocessor.links] # 嵌入一些代码，rust有用

# 第三方预处理器
[preprocessor.mathjax]

# 自定义预处理器
[preprocessor.random]
command = "python random.py"

# 渲染配置
[output.html]
mathjax-support = true # 公式支持
additional-css = ["custom.css"] # 自定义样式，一般用不着

[output.html.search]
limit-results = 15

```

其中多语言支持后的结构
``` bash
book/
├── en/  # 英文
│   ├── SUMMARY.md
│   ├── chapter_1.md
│   ├── chapter_2.md
│   └── ...
├── zh/  # 中文
│   ├── SUMMARY.md
│   ├── chapter_1.md
│   ├── chapter_2.md
│   └── ...
├── book.toml
└── ...
```


### 4. 公式

内联方程由 ```\\(``` 和 ```\\)``` 分隔，
块方程由 ```\\[``` 和 ```\\]``` 分隔。

\\[
 \frac{\partial c}{\partial t} = - \nabla \cdot \mathbf{v} c
\\]

### 5. 持续部署

GitHub Pages 对 mdbook 提高了非常好的支持。

点击 Settings => 点击 Pages => 在 source 中选择 GitHub Actions => 点击 mdbook 中的 Configure。

会自动生成一个 GitHub Actions 的配置文件 ``` .github/workflows/mdbook.yml ```,

不用修改，点击保存既可以，稍微等一会，就可以打开 GitHub Pages 提供的网站。

### 6. 经常忘记的 Markdown 语法

链接 和 图片

``` markdown

[mdBook](mdBook.md).
[mdBook](https://github.com/rust-lang/mdBook). 

![The Rust Logo](images/rust-logo-blk.svg)

```