# Jupyter Notebook

## 简介
Jupyter Notebook 是一个基于 Web 的交互式计算环境，可以用来创建和共享包含代码、方程、可视化和文本的文档。支持多种编程语言（如 Python、R、Julia 等）。

## 运行
```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

查看访问地址（默认需要token）
```
jupyter server list
```
用浏览器打开地址即可

## 常用快捷键

- **运行当前单元格并选中下一个**：`Shift + Enter`
- **运行当前单元格但保持选中状态**：`Ctrl + Enter`
- **中断代码运行**：`i i`（连按两次 i）
- **插入新单元格（在上方）**：`a` 或者 `A`
- **插入新单元格（在下方）**：`b`
- **删除当前单元格**：`D D`（连按两次 d）

- **Code（代码单元格）**
- **Markdown（文本单元格）**
