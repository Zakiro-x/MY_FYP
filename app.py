import os
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import predict as predict_tf

app = Flask(__name__)

# 上传文件保存目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 允许的图片后缀
ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 固定标签顺序
FIXED_ORDER = ["NonDemented", "VeryMildDementia", "MildDementia", "ModerateDementia"]

# 一个简单的页面模板（够毕设展示用）
HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AD MRI 智能识别系统</title>
  <!-- 引入 Google Fonts: Fredoka (可爱圆体) 和 Orbitron (科幻风格) -->
  <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@300;400;500;600&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
  <style>
    :root {
      /* 暗色背景 - 深空灰/黑 */
      --bg-color: #0f111a;
      /* 玻璃拟态卡片背景 */
      --card-bg: rgba(30, 32, 44, 0.75);
      /* 配色方案：赛博霓虹 + 可爱粉/紫 */
      --neon-blue: #00f3ff;
      --neon-purple: #bc13fe;
      --cute-pink: #ff55a3;
      --text-main: #e0e6ed;
      --text-sub: #949aa5;
    }
    
    * { box-sizing: border-box; }

    body {
      font-family: 'Fredoka', 'Microsoft YaHei', sans-serif;
      background-color: var(--bg-color);
      color: var(--text-main);
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      /* 背景光晕装饰 */
      background-image: 
        radial-gradient(circle at 15% 20%, rgba(188, 19, 254, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(0, 243, 255, 0.15) 0%, transparent 40%);
    }

    .container {
      width: 90%;
      max-width: 800px;
      padding: 40px;
      margin: 20px;
      border-radius: 30px; /* 大圆角体现可爱感 */
      background: var(--card-bg);
      backdrop-filter: blur(12px); /* 毛玻璃效果 */
      -webkit-backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
      border: 1px solid rgba(255, 255, 255, 0.08);
      text-align: center;
      position: relative;
    }

    h2 {
      font-family: 'Orbitron', sans-serif;
      font-weight: 700;
      font-size: 2rem;
      margin-bottom: 0.5rem;
      /* 渐变字 */
      background: linear-gradient(135deg, var(--neon-blue), var(--cute-pink));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: 1px;
    }

    p.subtitle {
      color: var(--text-sub);
      margin-top: 0;
      font-size: 1rem;
    }

    /* 装饰性小机器人晃动 */
    .mascot {
      font-size: 3.5rem;
      margin-bottom: 0px;
      animation: float 3s ease-in-out infinite;
      display: inline-block;
      filter: drop-shadow(0 0 10px var(--neon-purple));
    }
    @keyframes float {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }

    /* 上传区域 */
    .upload-box {
      margin-top: 2rem;
      background: rgba(255, 255, 255, 0.03);
      border: 2px dashed rgba(255, 85, 163, 0.4);
      border-radius: 20px;
      padding: 30px;
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .upload-box:hover {
      background: rgba(255, 85, 163, 0.08);
      border-color: var(--cute-pink);
    }
    
    /* 隐藏原生文件输入框 */
    input[type="file"] {
      display: none;
    }

    /* 自定义文件上传按钮 */
    .custom-file-upload {
      border: 1px solid var(--neon-blue);
      display: inline-block;
      padding: 12px 24px;
      cursor: pointer;
      border-radius: 50px;
      background: rgba(0, 243, 255, 0.1);
      color: var(--neon-blue);
      font-family: 'Fredoka', sans-serif;
      font-weight: 500;
      transition: all 0.3s ease;
      font-size: 1rem;
      margin-bottom: 20px;
      box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
    }

    .custom-file-upload:hover {
      background: var(--neon-blue);
      color: #0f111a; /* Dark background color */
      box-shadow: 0 0 20px rgba(0, 243, 255, 0.6);
    }
    
    .file-name-display {
      margin-top: 10px;
      color: var(--text-sub);
      font-size: 0.9rem;
      min-height: 1.2em; /* 占位避免跳动 */
    }
    
    .btn-submit {
      margin-top: 20px;
      background: linear-gradient(90deg, var(--neon-purple), var(--cute-pink));
      border: none;
      border-radius: 50px;
      padding: 12px 36px;
      color: #fff;
      font-size: 1.1rem;
      font-family: 'Fredoka', sans-serif;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(255, 85, 163, 0.4);
      transition: transform 0.2s;
    }
    .btn-submit:hover {
      transform: scale(1.05);
      box-shadow: 0 6px 20px rgba(255, 85, 163, 0.6);
    }

    /* 结果展示区 */
    .result-container {
      margin-top: 30px;
      padding-top: 20px;
      border-top: 1px solid rgba(255, 255, 255, 0.1);
      display: flex;
      flex-wrap: wrap;
      gap: 30px;
      text-align: left;
    }

    .img-box {
      flex: 1;
      min-width: 250px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }
    .img-box img {
      max-width: 100%;
      border-radius: 12px;
      box-shadow: 0 0 20px rgba(0, 243, 255, 0.2);
      border: 2px solid rgba(0, 243, 255, 0.3);
    }

    .info-box {
      flex: 1.5;
      min-width: 280px;
    }

    .badge {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 10px;
      font-weight: bold;
      margin-bottom: 10px;
    }
    .badge-pred {
      background: rgba(0, 243, 255, 0.15);
      color: var(--neon-blue);
      border: 1px solid var(--neon-blue);
      font-size: 1.3rem;
    }
    .badge-conf {
      background: rgba(188, 19, 254, 0.15);
      color: var(--neon-purple);
      border: 1px solid var(--neon-purple);
      margin-left: 10px;
    }

    /* 概率条样式 */
    .prob-row {
      display: flex;
      align-items: center;
      margin-bottom: 12px;
      font-size: 0.95rem;
    }
    .label-name {
      width: 120px;
      color: var(--text-sub);
    }
    .bar-container {
      flex: 1;
      height: 8px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 4px;
      margin: 0 10px;
      position: relative;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--neon-blue), var(--cute-pink));
      border-radius: 4px;
    }
    .prob-val {
      width: 50px;
      text-align: right;
      color: var(--text-main);
      font-weight: bold;
    }

    .warn {
      color: #ff5555;
      background: rgba(255, 85, 85, 0.1);
      padding: 10px;
      border-radius: 8px;
      margin-top: 15px;
    }

    .footer {
      margin-top: 2rem;
      font-size: 0.75rem;
      color: rgba(255, 255, 255, 0.3);
    }
  </style>
</head>
<body>

  <div class="container">
    <div class="mascot">🤖</div>
    <h2>阿尔茨海默病 AI 智能诊断</h2>
    <p class="subtitle">未来医疗助手 · 上传 MRI 切片进行分析</p>

    <form method="post" enctype="multipart/form-data" action="/predict">
      <div class="upload-box">
        <label for="file-upload" class="custom-file-upload">
          选择 MRI 影像文件
        </label>
        <input id="file-upload" type="file" name="file" accept="image/*" required onchange="updateFileName(this)"/>
        <div id="file-name" class="file-name-display">未选择任何文件</div>
        
        <button class="btn-submit" type="submit">开始预测 ✨</button>
      </div>
    </form>
    
    <script>
      function updateFileName(input) {
        var fileName = input.files[0] ? input.files[0].name : "未选择任何文件";
        document.getElementById("file-name").textContent = "已选择: " + fileName;
      }
    </script>

    {% if error %}
      <div class="warn">❌ {{ error }}</div>
    {% endif %}

    {% if result %}
      <div class="result-container">
        <!-- 左侧：图片 -->
        <div class="img-box">
          <img src="{{ result.image_url }}" alt="Upload">
        </div>

        <!-- 右侧：分析结果 -->
        <div class="info-box">
          <div>
            <div style="margin-bottom:5px; color:var(--text-sub); font-size:0.9rem;">预测结果 (Prediction)</div>
            <span class="badge badge-pred">{{ result.pred }}</span>
            <span class="badge badge-conf">置信度 {{ "%.1f"|format(result.conf * 100) }}%</span>
          </div>

          <div style="margin-top: 20px;">
             <div style="margin-bottom:10px; color:var(--text-sub); font-size:0.9rem;">概率分布详情：</div>
             {% for k in fixed_order %}
              <div class="prob-row">
                <span class="label-name">{{ k }}</span>
                <div class="bar-container">
                  <div class="bar-fill" style="width: {{ result.probs[k] * 100 }}%"></div>
                </div>
                <span class="prob-val">{{ "%.1f"|format(result.probs[k] * 100) }}%</span>
              </div>
             {% endfor %}
          </div>
          
          {% if result.conf < 0.60 %}
            <div style="margin-top:15px; font-size: 0.85rem; color: #ffb86c;">
              ⚠️ 提示：模型对该样本置信度较低，建议结合临床信息综合判断。
            </div>
          {% endif %}
        </div>
      </div>
    {% endif %}

    <div class="footer">
      Powered by 060822106 贺启玄 <br>
      System ready: 2026.02.07
    </div>
  </div>

</body>
</html>
"""


def allowed_file(filename: str) -> bool:
    name = filename.lower()
    return any(name.endswith(ext) for ext in ALLOWED_EXTS)


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, result=None, error=None, fixed_order=FIXED_ORDER)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML, result=None, error="未检测到上传文件。", fixed_order=FIXED_ORDER)

    f = request.files["file"]
    if f.filename == "":
        return render_template_string(HTML, result=None, error="文件名为空，请重新选择图片。", fixed_order=FIXED_ORDER)

    if not allowed_file(f.filename):
        return render_template_string(
            HTML,
            result=None,
            error=f"不支持的文件类型：{f.filename}（请上传 {', '.join(ALLOWED_EXTS)}）",
            fixed_order=FIXED_ORDER
        )

    # 保存上传文件
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)

    # 获取缓存的模型与标签映射（第一次调用会加载，之后复用）
    model, labels = predict_tf.get_model_and_labels()

    # 自动适配模型输入尺寸（避免 224/300 再踩坑）
    img_size = model.input_shape[1]

    # 推理
    pred, conf, dist = predict_tf.predict_one(model, labels, save_path, img_size=img_size)

    # 给浏览器展示图片（用相对路径）
    image_url = f"/uploads/{filename}"

    result = {
        "pred": pred,
        "conf": conf,
        "probs": dist,
        "image_url": image_url,
    }
    return render_template_string(HTML, result=result, error=None, fixed_order=FIXED_ORDER)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # 简单静态文件服务：让页面能显示上传的图片
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    # 本机演示：浏览器打开 http://127.0.0.1:5000
    # debug=True 方便你开发期看报错；演示时也可以保持 True
    app.run(host="127.0.0.1", port=5000, debug=True)