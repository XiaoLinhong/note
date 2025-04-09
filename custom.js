document.addEventListener('DOMContentLoaded', function () {
    let headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    let sectionCounters = [1, 0, 0, 0, 0, 0];  // 每个级别的计数器，初始值为0

    headings.forEach(function (heading, index) {
        let level = parseInt(heading.nodeName[1]);

        // 跳过 h1
        if (level === 1) {
            return; // 跳过标题，不做任何修改
        }

        // 增加当前层级的计数器
        sectionCounters[level - 1]++;

        // 将更高层级的计数器重置为下一级的计数
        for (let i = level; i < sectionCounters.length; i++) {
            sectionCounters[i] = 0;
        }

        // 构造序号
        let prefix = sectionCounters.slice(1, level).join('.') + '.';
        let headingText = heading.innerText;

        // 更新标题内容
        heading.innerText = `${prefix} ${headingText}`;
    });
});
