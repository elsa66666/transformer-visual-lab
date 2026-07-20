# Transformer 结构实验室

一个面向 Transformer 初学者的交互式结构可视化网页。

## 功能

- 2D 教学流程图与 3D 分层透视图一键切换
- 每个模块均支持悬停名称提示和点击详情
- Self-Attention 展开到 `Q / K / V`、`W_Q / W_K / W_V / W_O`
- FFN 展开到 `W_in / GELU / W_out`
- 包含公式、张量形状、大白话类比和运行示例
- 支持 Decoder-only 与 Encoder–Decoder 视角切换
- 键盘可访问，并适配窄屏浏览

## 本地预览

无需安装依赖，直接打开 `docs/index.html`，或在仓库根目录运行：

```bash
python3 -m http.server 8000 --directory docs
```

然后访问 `http://localhost:8000`。

## GitHub Pages

仓库包含 `.github/workflows/pages.yml`。推送到 `main` 后，工作流会把 `docs/` 发布到 GitHub Pages。

首次使用时，在仓库的 **Settings → Pages → Build and deployment** 中将 Source 设为 **GitHub Actions**。

## 项目结构

```text
docs/
├── index.html    # 页面结构
├── styles.css    # 视觉样式与 2D/3D 效果
└── app.js        # 交互逻辑与教学内容
```

## License

MIT
