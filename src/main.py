import asyncio
import json
import re
import uuid
import time
import secrets
import base64
import mimetypes
from collections import defaultdict
from typing import Optional, Dict, List
from datetime import datetime, timezone, timedelta

import uvicorn
from camoufox.async_api import AsyncCamoufox
from fastapi import FastAPI, HTTPException, Depends, status, Form, Request, Response
from starlette.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import APIKeyHeader

import httpx

# ============================================================
# 配置
# ============================================================
# 设置为 True 以进行详细日志记录，设置为 False 以进行最少日志记录
DEBUG = False

# 运行服务器的端口
PORT = 8000
# ============================================================

def debug_print(*args, **kwargs):
    """仅在 DEBUG 为 True 时打印调试消息"""
    if DEBUG:
        print(*args, **kwargs)

# 自定义 UUIDv7 实现（使用正确的 Unix 纪元）
def uuid7():
    """
    使用 Unix 纪元（自 1970-01-01 以来的毫秒数）生成 UUIDv7
    与浏览器的实现相匹配。
    """
    timestamp_ms = int(time.time() * 1000)
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)
    
    uuid_int = timestamp_ms << 80
    uuid_int |= (0x7000 | rand_a) << 64
    uuid_int |= (0x8000000000000000 | rand_b)
    
    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"

# 图片上传辅助函数
async def upload_image_to_lmarena(image_data: bytes, mime_type: str, filename: str) -> Optional[tuple]:
    """
    上传图片到 LMArena R2 存储并返回密钥和下载 URL。
    
    参数:
        image_data: 二进制图片数据
        mime_type: 图片的 MIME 类型 (例如 'image/png')
        filename: 图片的原始文件名
    
    返回:
        如果成功，返回 (key, download_url) 元组，如果上传失败则返回 None
    """
    try:
        # 验证输入
        if not image_data:
            debug_print("❌ 图片数据为空")
            return None
        
        if not mime_type or not mime_type.startswith('image/'):
            debug_print(f"❌ 无效的 MIME 类型: {mime_type}")
            return None
        
        # 步骤 1: 请求上传 URL
        debug_print(f"📤 步骤 1: 请求 {filename} 的上传 URL")
        
        # 为 Next.js Server Action 准备标头
        request_headers = get_request_headers()
        request_headers.update({
            "Accept": "text/x-component",
            "Content-Type": "text/plain;charset=UTF-8",
            "Next-Action": "70cb393626e05a5f0ce7dcb46977c36c139fa85f91",
            "Referer": "https://lmarena.ai/?mode=direct",
        })
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://lmarena.ai/?mode=direct",
                    headers=request_headers,
                    content=json.dumps([filename, mime_type]),
                    timeout=30.0
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                debug_print("❌ 请求上传 URL 超时")
                return None
            except httpx.HTTPError as e:
                debug_print(f"❌ 请求上传 URL 时发生 HTTP 错误: {e}")
                return None
            
            # 解析响应 - 格式: 0:{...}\n1:{...}\n
            try:
                lines = response.text.strip().split('\n')
                upload_data = None
                for line in lines:
                    if line.startswith('1:'):
                        upload_data = json.loads(line[2:])
                        break
                
                if not upload_data or not upload_data.get('success'):
                    debug_print(f"❌ 获取上传 URL 失败: {response.text[:200]}")
                    return None
                
                upload_url = upload_data['data']['uploadUrl']
                key = upload_data['data']['key']
                debug_print(f"✅ 获取到上传 URL 和密钥: {key}")
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                debug_print(f"❌ 解析上传 URL 响应失败: {e}")
                return None
            
            # 步骤 2: 上传图片到 R2 存储
            debug_print(f"📤 步骤 2: 上传图片到 R2 存储 ({len(image_data)} 字节)")
            try:
                response = await client.put(
                    upload_url,
                    content=image_data,
                    headers={"Content-Type": mime_type},
                    timeout=60.0
                )
                response.raise_for_status()
                debug_print(f"✅ 图片上传成功")
            except httpx.TimeoutException:
                debug_print("❌ 上传图片到 R2 存储超时")
                return None
            except httpx.HTTPError as e:
                debug_print(f"❌ 上传图片时发生 HTTP 错误: {e}")
                return None
            
            # 步骤 3: 获取签名下载 URL (使用不同的 Next-Action)
            debug_print(f"📤 步骤 3: 请求签名下载 URL")
            request_headers_step3 = request_headers.copy()
            request_headers_step3["Next-Action"] = "6064c365792a3eaf40a60a874b327fe031ea6f22d7"
            
            try:
                response = await client.post(
                    "https://lmarena.ai/?mode=direct",
                    headers=request_headers_step3,
                    content=json.dumps([key]),
                    timeout=30.0
                )
                response.raise_for_status()
            except httpx.TimeoutException:
                debug_print("❌ 请求下载 URL 超时")
                return None
            except httpx.HTTPError as e:
                debug_print(f"❌ 请求下载 URL 时发生 HTTP 错误: {e}")
                return None
            
            # 解析响应
            try:
                lines = response.text.strip().split('\n')
                download_data = None
                for line in lines:
                    if line.startswith('1:'):
                        download_data = json.loads(line[2:])
                        break
                
                if not download_data or not download_data.get('success'):
                    debug_print(f"❌ 获取下载 URL 失败: {response.text[:200]}")
                    return None
                
                download_url = download_data['data']['url']
                debug_print(f"✅ 获取到签名下载 URL: {download_url[:100]}...")
                return (key, download_url)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                debug_print(f"❌ 解析下载 URL 响应失败: {e}")
                return None
            
    except Exception as e:
        debug_print(f"❌ 上传图片时发生意外错误: {type(e).__name__}: {e}")
        return None

async def process_message_content(content, model_capabilities: dict) -> tuple[str, List[dict]]:
    """
    处理消息内容，如果存在图片且模型支持，则处理图片。
    
    参数:
        content: 消息内容 (字符串或内容部分列表)
        model_capabilities: 模型的 capability 字典
    
    返回:
        (text_content, experimental_attachments) 元组
    """
    # 检查模型是否支持图片输入
    supports_images = model_capabilities.get('inputCapabilities', {}).get('image', False)
    
    # 如果内容是字符串，则按原样返回
    if isinstance(content, str):
        return content, []
    
    # 如果内容是列表 (OpenAI 格式，包含多个部分)
    if isinstance(content, list):
        text_parts = []
        attachments = []
        
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
                    
                elif part.get('type') == 'image_url' and supports_images:
                    image_url = part.get('image_url', {})
                    if isinstance(image_url, dict):
                        url = image_url.get('url', '')
                    else:
                        url = image_url
                    
                    # 处理 base64 编码的图片
                    if url.startswith('data:'):
                        # 格式: data:image/png;base64,iVBORw0KGgo...
                        try:
                            # 验证并解析数据 URI
                            if ',' not in url:
                                debug_print(f"❌ 无效的数据 URI 格式（无逗号分隔符）")
                                continue
                            
                            header, data = url.split(',', 1)
                            
                            # 解析 MIME 类型
                            if ';' not in header or ':' not in header:
                                debug_print(f"❌ 无效的数据 URI 标头格式")
                                continue
                            
                            mime_type = header.split(';')[0].split(':')[1]
                            
                            # 验证 MIME 类型
                            if not mime_type.startswith('image/'):
                                debug_print(f"❌ 无效的 MIME 类型: {mime_type}")
                                continue
                            
                            # 解码 base64
                            try:
                                image_data = base64.b64decode(data)
                            except Exception as e:
                                debug_print(f"❌ 解码 base64 数据失败: {e}")
                                continue
                            
                            # 验证图片大小 (最大 10MB)
                            if len(image_data) > 10 * 1024 * 1024:
                                debug_print(f"❌ 图片过大: {len(image_data)} 字节 (最大 10MB)")
                                continue
                            
                            # 生成文件名
                            ext = mimetypes.guess_extension(mime_type) or '.png'
                            filename = f"upload-{uuid.uuid4()}{ext}"
                            
                            debug_print(f"🖼️  处理 base64 图片: {filename}, 大小: {len(image_data)} 字节")
                            
                            # 上传到 LMArena
                            upload_result = await upload_image_to_lmarena(image_data, mime_type, filename)
                            
                            if upload_result:
                                key, download_url = upload_result
                                # 添加为 LMArena 格式的附件
                                attachments.append({
                                    "name": key,
                                    "contentType": mime_type,
                                    "url": download_url
                                })
                                debug_print(f"✅ 图片已上传并添加到附件")
                            else:
                                debug_print(f"⚠️  上传图片失败，跳过")
                        except Exception as e:
                            debug_print(f"❌ 处理 base64 图片时发生意外错误: {type(e).__name__}: {e}")
                    
                    # 处理 URL 图片 (直接 URL)
                    elif url.startswith('http://') or url.startswith('https://'):
                        # 对于外部 URL，我们需要下载并重新上传
                        # 目前跳过此情况
                        debug_print(f"⚠️  尚不支持外部图片 URL: {url[:100]}")
                        
                elif part.get('type') == 'image_url' and not supports_images:
                    debug_print(f"⚠️  提供了图片，但模型不支持图片")
        
        # 合并文本部分
        text_content = '\n'.join(text_parts).strip()
        return text_content, attachments
    
    # 回退
    return str(content), []

app = FastAPI()

# --- 常量和全局状态 ---
CONFIG_FILE = "config.json"
MODELS_FILE = "models.json"
API_KEY_HEADER = APIKeyHeader(name="Authorization")

# 内存存储
# { "api_key": { "conversation_id": session_data } }
chat_sessions: Dict[str, Dict[str, dict]] = defaultdict(dict)
# { "session_id": "username" }
dashboard_sessions = {}
# { "api_key": [timestamp1, timestamp2, ...] }
api_key_usage = defaultdict(list)
# { "model_id": count }
model_usage_stats = defaultdict(int)

# --- 辅助函数 ---

def get_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}

    # 确保默认键存在
    config.setdefault("password", "admin")
    config.setdefault("auth_token", "")
    config.setdefault("cf_clearance", "")
    config.setdefault("api_keys", [])
    config.setdefault("usage_stats", {})
    
    return config

def load_usage_stats():
    """从配置加载使用统计到内存"""
    global model_usage_stats
    config = get_config()
    model_usage_stats = defaultdict(int, config.get("usage_stats", {}))

def save_config(config):
    # 保存前将内存中的统计数据持久化到配置字典
    config["usage_stats"] = dict(model_usage_stats)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def get_models():
    try:
        with open(MODELS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_models(models):
    with open(MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)

def get_request_headers():
    config = get_config()
    auth_token = config.get("auth_token", "").strip()
    if not auth_token:
        raise HTTPException(status_code=500, detail="仪表板中未设置 Arena 认证令牌。")
    
    cf_clearance = config.get("cf_clearance", "").strip()
    return {
        "Content-Type": "application/json",
        "Cookie": f"cf_clearance={cf_clearance}; arena-auth-prod-v1={auth_token}",
    }

# --- 仪表板认证 ---

async def get_current_session(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in dashboard_sessions:
        return dashboard_sessions[session_id]
    return None

# --- API 密钥认证和速率限制 ---

async def rate_limit_api_key(key: str = Depends(API_KEY_HEADER)):
    if not key.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="无效的 Authorization 标头。应为 'Bearer YOUR_API_KEY'"
        )
    
    # 移除 "Bearer " 前缀并去除空格
    api_key_str = key[7:].strip()
    config = get_config()
    
    key_data = next((k for k in config["api_keys"] if k["key"] == api_key_str), None)
    if not key_data:
        raise HTTPException(status_code=401, detail="无效的 API 密钥。")

    # 速率限制
    rate_limit = key_data.get("rpm", 60)
    current_time = time.time()
    
    # 清理旧的时间戳 (超过 60 秒)
    api_key_usage[api_key_str] = [t for t in api_key_usage[api_key_str] if current_time - t < 60]

    if len(api_key_usage[api_key_str]) >= rate_limit:
        # 计算直到最旧请求过期的时间 (60 秒窗口)
        oldest_timestamp = min(api_key_usage[api_key_str])
        retry_after = int(60 - (current_time - oldest_timestamp))
        retry_after = max(1, retry_after)  # 至少 1 秒
        
        raise HTTPException(
            status_code=429,
            detail="超出速率限制。请稍后再试。",
            headers={"Retry-After": str(retry_after)}
        )
        
    api_key_usage[api_key_str].append(current_time)
    
    return key_data

# --- 核心逻辑 ---

async def get_initial_data():
    print("开始初始数据获取...")
    try:
        async with AsyncCamoufox(headless=True) as browser:
            page = await browser.new_page()
            
            print("正在导航至 lmarena.ai...")
            await page.goto("https://lmarena.ai/", wait_until="domcontentloaded")

            print("正在等待 Cloudflare 验证完成...")
            try:
                await page.wait_for_function(
                    "() => document.title.indexOf('Just a moment...') === -1", 
                    timeout=45000
                )
                print("✅ Cloudflare 验证通过。")
            except Exception as e:
                print(f"❌ Cloudflare 验证耗时过长或失败: {e}")
                return

            await asyncio.sleep(5)

            # 提取 cf_clearance
            cookies = await page.context.cookies()
            cf_clearance_cookie = next((c for c in cookies if c["name"] == "cf_clearance"), None)
            
            config = get_config()
            if cf_clearance_cookie:
                config["cf_clearance"] = cf_clearance_cookie["value"]
                save_config(config)
                print(f"✅ 已保存 cf_clearance 令牌: {cf_clearance_cookie['value'][:20]}...")
            else:
                print("⚠️ 找不到 cf_clearance cookie。")

            # 提取模型
            print("正在从页面提取模型...")
            try:
                body = await page.content()
                match = re.search(r'{\\"initialModels\\":(\[.*?\]),\\"initialModel[A-Z]Id', body, re.DOTALL)
                if match:
                    models_json = match.group(1).encode().decode('unicode_escape')
                    models = json.loads(models_json)
                    save_models(models)
                    print(f"✅ 已保存 {len(models)} 个模型")
                else:
                    print("⚠️ 页面中找不到模型")
            except Exception as e:
                print(f"❌ 提取模型时出错: {e}")

            print("✅ 初始数据获取完成")
    except Exception as e:
        print(f"❌ 初始数据获取期间发生错误: {e}")

async def periodic_refresh_task():
    """后台任务：每 30 分钟刷新 cf_clearance 和模型"""
    while True:
        try:
            # 等待 30 分钟 (1800 秒)
            await asyncio.sleep(1800)
            print("\n" + "="*60)
            print("🔄 开始计划的 30 分钟刷新...")
            print("="*60)
            await get_initial_data()
            print("✅ 计划刷新完成")
            print("="*60 + "\n")
        except Exception as e:
            print(f"❌ 定期刷新任务出错: {e}")
            # 即使出错也继续循环
            continue

@app.on_event("startup")
async def startup_event():
    # 确保配置和模型文件存在
    save_config(get_config())
    save_models(get_models())
    # 从配置加载使用统计
    load_usage_stats()
    # 启动初始数据获取
    asyncio.create_task(get_initial_data())
    # 启动定期刷新任务 (每 30 分钟)
    asyncio.create_task(periodic_refresh_task())

# --- UI 端点 (登录/仪表板) ---

@app.get("/", response_class=HTMLResponse)
async def root_redirect():
    return RedirectResponse(url="/dashboard")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    if await get_current_session(request):
        return RedirectResponse(url="/dashboard")
    
    error_msg = '<div class="error-message">密码无效。请重试。</div>' if error else ''
    
    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>登录 - LMArena Bridge</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .login-container {{
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    width: 100%;
                    max-width: 400px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 10px;
                    font-size: 28px;
                }}
                .subtitle {{
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 14px;
                }}
                .form-group {{
                    margin-bottom: 20px;
                }}
                label {{
                    display: block;
                    margin-bottom: 8px;
                    color: #555;
                    font-weight: 500;
                }}
                input[type="password"] {{
                    width: 100%;
                    padding: 12px;
                    border: 2px solid #e1e8ed;
                    border-radius: 6px;
                    font-size: 16px;
                    transition: border-color 0.3s;
                }}
                input[type="password"]:focus {{
                    outline: none;
                    border-color: #667eea;
                }}
                button {{
                    width: 100%;
                    padding: 12px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: transform 0.2s;
                }}
                button:hover {{
                    transform: translateY(-2px);
                }}
                button:active {{
                    transform: translateY(0);
                }}
                .error-message {{
                    background: #fee;
                    color: #c33;
                    padding: 12px;
                    border-radius: 6px;
                    margin-bottom: 20px;
                    border-left: 4px solid #c33;
                }}
            </style>
        </head>
        <body>
            <div class="login-container">
                <h1>LMArena Bridge</h1>
                <div class="subtitle">登录以访问仪表板</div>
                {error_msg}
                <form action="/login" method="post">
                    <div class="form-group">
                        <label for="password">密码</label>
                        <input type="password" id="password" name="password" placeholder="输入您的密码" required autofocus>
                    </div>
                    <button type="submit">登录</button>
                </form>
            </div>
        </body>
        </html>
    """

@app.post("/login")
async def login_submit(response: Response, password: str = Form(...)):
    config = get_config()
    if password == config.get("password"):
        session_id = str(uuid.uuid4())
        dashboard_sessions[session_id] = "admin"
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response
    return RedirectResponse(url="/login?error=1", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id in dashboard_sessions:
        del dashboard_sessions[session_id]
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("session_id")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(session: str = Depends(get_current_session)):
    if not session:
        return RedirectResponse(url="/login")

    config = get_config()
    models = get_models()

    # 渲染 API 密钥
    keys_html = ""
    for key in config["api_keys"]:
        created_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(key.get('created', 0)))
        keys_html += f"""
            <tr>
                <td><strong>{key['name']}</strong></td>
                <td><code class="api-key-code">{key['key']}</code></td>
                <td><span class="badge">{key['rpm']} RPM</span></td>
                <td><small>{created_date}</small></td>
                <td>
                    <form action='/delete-key' method='post' style='margin:0;' onsubmit='return confirm("删除此 API 密钥？");'>
                        <input type='hidden' name='key_id' value='{key['key']}'>
                        <button type='submit' class='btn-delete'>删除</button>
                    </form>
                </td>
            </tr>
        """

    # 渲染模型（限制前 20 个具有文本输出的模型）
    text_models = [m for m in models if m.get('capabilities', {}).get('outputCapabilities', {}).get('text')]
    models_html = ""
    for i, model in enumerate(text_models[:20]):
        rank = model.get('rank', '?')
        org = model.get('organization', 'Unknown')
        models_html += f"""
            <div class="model-card">
                <div class="model-header">
                    <span class="model-name">{model.get('publicName', 'Unnamed')}</span>
                    <span class="model-rank">排名 {rank}</span>
                </div>
                <div class="model-org">{org}</div>
            </div>
        """
    
    if not models_html:
        models_html = '<div class="no-data">未找到模型。令牌可能无效或已过期。</div>'

    # 渲染统计数据
    stats_html = ""
    if model_usage_stats:
        for model, count in sorted(model_usage_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            stats_html += f"<tr><td>{model}</td><td><strong>{count}</strong></td></tr>"
    else:
        stats_html = "<tr><td colspan='2' class='no-data'>暂无使用数据</td></tr>"

    # 检查令牌状态
    token_status = "✅ 已配置" if config.get("auth_token") else "❌ 未设置"
    token_class = "status-good" if config.get("auth_token") else "status-bad"
    
    cf_status = "✅ 已配置" if config.get("cf_clearance") else "❌ 未设置"
    cf_class = "status-good" if config.get("cf_clearance") else "status-bad"
    
    # 获取最近活动计数（过去 24 小时）
    recent_activity = sum(1 for timestamps in api_key_usage.values() for t in timestamps if time.time() - t < 86400)

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard - LMArena Bridge</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
            <style>
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                @keyframes slideIn {{
                    from {{ opacity: 0; transform: translateX(-20px); }}
                    to {{ opacity: 1; transform: translateX(0); }}
                }}
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); }}
                    50% {{ transform: scale(1.05); }}
                }}
                @keyframes shimmer {{
                    0% {{ background-position: -1000px 0; }}
                    100% {{ background-position: 1000px 0; }}
                }}
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: #f5f7fa;
                    color: #333;
                    line-height: 1.6;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header-content {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 0 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                h1 {{
                    font-size: 24px;
                    font-weight: 600;
                }}
                .logout-btn {{
                    background: rgba(255,255,255,0.2);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 6px;
                    text-decoration: none;
                    transition: background 0.3s;
                }}
                .logout-btn:hover {{
                    background: rgba(255,255,255,0.3);
                }}
                .container {{
                    max-width: 1200px;
                    margin: 30px auto;
                    padding: 0 20px;
                }}
                .section {{
                    background: white;
                    border-radius: 10px;
                    padding: 25px;
                    margin-bottom: 25px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }}
                .section-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #f0f0f0;
                }}
                h2 {{
                    font-size: 20px;
                    color: #333;
                    font-weight: 600;
                }}
                .status-badge {{
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: 600;
                }}
                .status-good {{ background: #d4edda; color: #155724; }}
                .status-bad {{ background: #f8d7da; color: #721c24; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th {{
                    background: #f8f9fa;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #555;
                    font-size: 14px;
                    border-bottom: 2px solid #e9ecef;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #f0f0f0;
                }}
                tr:hover {{
                    background: #f8f9fa;
                }}
                .form-group {{
                    margin-bottom: 15px;
                }}
                label {{
                    display: block;
                    margin-bottom: 6px;
                    font-weight: 500;
                    color: #555;
                }}
                input[type="text"], input[type="number"], textarea {{
                    width: 100%;
                    padding: 10px;
                    border: 2px solid #e1e8ed;
                    border-radius: 6px;
                    font-size: 14px;
                    font-family: inherit;
                    transition: border-color 0.3s;
                }}
                input:focus, textarea:focus {{
                    outline: none;
                    border-color: #667eea;
                }}
                textarea {{
                    resize: vertical;
                    font-family: 'Courier New', monospace;
                    min-height: 100px;
                }}
                button, .btn {{
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s;
                }}
                button[type="submit"]:not(.btn-delete) {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                button[type="submit"]:not(.btn-delete):hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                }}
                .btn-delete {{
                    background: #dc3545;
                    color: white;
                    padding: 6px 12px;
                    font-size: 13px;
                }}
                .btn-delete:hover {{
                    background: #c82333;
                }}
                .api-key-code {{
                    background: #f8f9fa;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    color: #495057;
                }}
                .badge {{
                    background: #e7f3ff;
                    color: #0066cc;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                .model-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .model-card {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .model-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }}
                .model-name {{
                    font-weight: 600;
                    color: #333;
                    font-size: 14px;
                }}
                .model-rank {{
                    background: #667eea;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 12px;
                    font-size: 11px;
                    font-weight: 600;
                }}
                .model-org {{
                    color: #666;
                    font-size: 12px;
                }}
                .no-data {{
                    text-align: center;
                    color: #999;
                    padding: 20px;
                    font-style: italic;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    animation: fadeIn 0.6s ease-out;
                    transition: transform 0.3s;
                }}
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
                }}
                .section {{
                    animation: slideIn 0.5s ease-out;
                }}
                .section:nth-child(2) {{ animation-delay: 0.1s; }}
                .section:nth-child(3) {{ animation-delay: 0.2s; }}
                .section:nth-child(4) {{ animation-delay: 0.3s; }}
                .model-card {{
                    animation: fadeIn 0.4s ease-out;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .model-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .form-row {{
                    display: grid;
                    grid-template-columns: 2fr 1fr auto;
                    gap: 10px;
                    align-items: end;
                }}
                @media (max-width: 768px) {{
                    .form-row {{
                        grid-template-columns: 1fr;
                    }}
                    .model-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="header-content">
                    <h1>🚀 LMArena Bridge 仪表板</h1>
                    <a href="/logout" class="logout-btn">退出登录</a>
                </div>
            </div>

            <div class="container">
                <!-- Stats Overview -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{len(config['api_keys'])}</div>
                        <div class="stat-label">API 密钥</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(text_models)}</div>
                        <div class="stat-label">可用模型</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(model_usage_stats.values())}</div>
                        <div class="stat-label">总请求数</div>
                    </div>
                </div>

                <!-- Arena Auth Token -->
                <div class="section">
                    <div class="section-header">
                        <h2>🔐 Arena 认证</h2>
                        <span class="status-badge {token_class}">{token_status}</span>
                    </div>
                    <form action="/update-auth-token" method="post">
                        <div class="form-group">
                            <label for="auth_token">Arena 认证令牌</label>
                            <textarea id="auth_token" name="auth_token" placeholder="在此粘贴您的 arena-auth-prod-v1 令牌">{config.get("auth_token", "")}</textarea>
                        </div>
                        <button type="submit">更新令牌</button>
                    </form>
                </div>

                <!-- Cloudflare Clearance -->
                <div class="section">
                    <div class="section-header">
                        <h2>☁️ Cloudflare Clearance</h2>
                        <span class="status-badge {cf_class}">{cf_status}</span>
                    </div>
                    <p style="color: #666; margin-bottom: 15px;">这是在启动时自动获取的。如果 API 请求失败并出现 404 错误，则令牌可能已过期。</p>
                    <code style="background: #f8f9fa; padding: 10px; display: block; border-radius: 6px; word-break: break-all; margin-bottom: 15px;">
                        {config.get("cf_clearance", "未设置")}
                    </code>
                    <form action="/refresh-tokens" method="post" style="margin-top: 15px;">
                        <button type="submit" style="background: #28a745;">🔄 刷新令牌和模型</button>
                    </form>
                    <p style="color: #999; font-size: 13px; margin-top: 10px;"><em>注意：这将获取新的 cf_clearance 令牌并更新模型列表。</em></p>
                </div>

                <!-- API Keys -->
                <div class="section">
                    <div class="section-header">
                        <h2>🔑 API 密钥</h2>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>名称</th>
                                <th>密钥</th>
                                <th>速率限制</th>
                                <th>创建时间</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            {keys_html if keys_html else '<tr><td colspan="5" class="no-data">未配置 API 密钥</td></tr>'}
                        </tbody>
                    </table>
                    
                    <h3 style="margin-top: 30px; margin-bottom: 15px; font-size: 18px;">创建新 API 密钥</h3>
                    <form action="/create-key" method="post">
                        <div class="form-row">
                            <div class="form-group">
                                <label for="name">密钥名称</label>
                                <input type="text" id="name" name="name" placeholder="例如：生产密钥" required>
                            </div>
                            <div class="form-group">
                                <label for="rpm">速率限制 (RPM)</label>
                                <input type="number" id="rpm" name="rpm" value="60" min="1" max="1000" required>
                            </div>
                            <div class="form-group">
                                <label>&nbsp;</label>
                                <button type="submit">创建密钥</button>
                            </div>
                        </div>
                    </form>
                </div>

                <!-- Usage Statistics -->
                <div class="section">
                    <div class="section-header">
                        <h2>📊 使用统计</h2>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px;">
                        <div>
                            <h3 style="text-align: center; margin-bottom: 15px; font-size: 16px; color: #666;">模型使用分布</h3>
                            <canvas id="modelPieChart" style="max-height: 300px;"></canvas>
                        </div>
                        <div>
                            <h3 style="text-align: center; margin-bottom: 15px; font-size: 16px; color: #666;">按模型的请求计数</h3>
                            <canvas id="modelBarChart" style="max-height: 300px;"></canvas>
                        </div>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>模型</th>
                                <th>请求数</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stats_html}
                        </tbody>
                    </table>
                </div>

                <!-- Available Models -->
                <div class="section">
                    <div class="section-header">
                        <h2>🤖 可用模型</h2>
                    </div>
                    <p style="color: #666; margin-bottom: 15px;">显示前 20 个基于文本的模型（排名 1 = 最佳）</p>
                    <div class="model-grid">
                        {models_html}
                    </div>
                </div>
            </div>
            
            <script>
                // Prepare data for charts
                const statsData = {json.dumps(dict(sorted(model_usage_stats.items(), key=lambda x: x[1], reverse=True)[:10]))};
                const modelNames = Object.keys(statsData);
                const modelCounts = Object.values(statsData);
                
                // Generate colors for charts
                const colors = [
                    '#667eea', '#764ba2', '#f093fb', '#4facfe',
                    '#43e97b', '#fa709a', '#fee140', '#30cfd0',
                    '#a8edea', '#fed6e3'
                ];
                
                // Pie Chart
                if (modelNames.length > 0) {{
                    const pieCtx = document.getElementById('modelPieChart').getContext('2d');
                    new Chart(pieCtx, {{
                        type: 'doughnut',
                        data: {{
                            labels: modelNames,
                            datasets: [{{
                                data: modelCounts,
                                backgroundColor: colors,
                                borderWidth: 2,
                                borderColor: '#fff'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {{
                                legend: {{
                                    position: 'bottom',
                                    labels: {{
                                        padding: 15,
                                        font: {{
                                            size: 11
                                        }}
                                    }}
                                }},
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const label = context.label || '';
                                            const value = context.parsed || 0;
                                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                            const percentage = ((value / total) * 100).toFixed(1);
                                            return label + ': ' + value + ' (' + percentage + '%)';
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                    
                    // Bar Chart
                    const barCtx = document.getElementById('modelBarChart').getContext('2d');
                    new Chart(barCtx, {{
                        type: 'bar',
                        data: {{
                            labels: modelNames,
                            datasets: [{{
                                label: '请求数',
                                data: modelCounts,
                                backgroundColor: colors[0],
                                borderColor: colors[1],
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {{
                                legend: {{
                                    display: false
                                }},
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            return '请求数: ' + context.parsed.y;
                                        }}
                                    }}
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    ticks: {{
                                        stepSize: 1
                                    }}
                                }},
                                x: {{
                                    ticks: {{
                                        font: {{
                                            size: 10
                                        }},
                                        maxRotation: 45,
                                        minRotation: 45
                                    }}
                                }}
                            }}
                        }}
                    }});
                }} else {{
                    // Show "no data" message
                    document.getElementById('modelPieChart').parentElement.innerHTML = '<p style="text-align: center; color: #999; padding: 50px;">暂无使用数据</p>';
                    document.getElementById('modelBarChart').parentElement.innerHTML = '<p style="text-align: center; color: #999; padding: 50px;">暂无使用数据</p>';
                }}
            </script>
        </body>
        </html>
    """

@app.post("/update-auth-token")
async def update_auth_token(session: str = Depends(get_current_session), auth_token: str = Form(...)):
    if not session:
        return RedirectResponse(url="/login")
    config = get_config()
    config["auth_token"] = auth_token.strip()
    save_config(config)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/create-key")
async def create_key(session: str = Depends(get_current_session), name: str = Form(...), rpm: int = Form(...)):
    if not session:
        return RedirectResponse(url="/login")
    config = get_config()
    new_key = {
        "name": name.strip(),
        "key": f"sk-lmab-{uuid.uuid4()}",
        "rpm": max(1, min(rpm, 1000)),  # Clamp between 1-1000
        "created": int(time.time())
    }
    config["api_keys"].append(new_key)
    save_config(config)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/delete-key")
async def delete_key(session: str = Depends(get_current_session), key_id: str = Form(...)):
    if not session:
        return RedirectResponse(url="/login")
    config = get_config()
    config["api_keys"] = [k for k in config["api_keys"] if k["key"] != key_id]
    save_config(config)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/refresh-tokens")
async def refresh_tokens(session: str = Depends(get_current_session)):
    if not session:
        return RedirectResponse(url="/login")
    await get_initial_data()
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

# --- OpenAI 兼容 API 端点 ---

@app.get("/api/v1/health")
async def health_check():
    """用于监控的健康检查端点"""
    try:
        models = get_models()
        config = get_config()
        
        # 基本健康检查
        has_cf_clearance = bool(config.get("cf_clearance"))
        has_models = len(models) > 0
        has_api_keys = len(config.get("api_keys", [])) > 0
        
        status = "healthy" if (has_cf_clearance and has_models) else "degraded"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "cf_clearance": has_cf_clearance,
                "models_loaded": has_models,
                "model_count": len(models),
                "api_keys_configured": has_api_keys
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@app.get("/api/v1/models")
async def list_models(api_key: dict = Depends(rate_limit_api_key)):
    models = get_models()
    # 过滤具有文本或搜索输出能力且有组织（排除隐形模型）的模型
    # 包括聊天、搜索和 Web 开发模型
    valid_models = [m for m in models 
                   if (m.get('capabilities', {}).get('outputCapabilities', {}).get('text')
                       or m.get('capabilities', {}).get('outputCapabilities', {}).get('search'))
                   and m.get('organization')]
    
    return {
        "object": "list",
        "data": [
            {
                "id": model.get("publicName"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": model.get("organization", "lmarena")
            } for model in valid_models if model.get("publicName")
        ]
    }

@app.post("/api/v1/chat/completions")
async def api_chat_completions(request: Request, api_key: dict = Depends(rate_limit_api_key)):
    debug_print("\n" + "="*80)
    debug_print("🔵 收到新的 API 请求")
    debug_print("="*80)
    
    try:
        # 解析请求体并处理错误
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            debug_print(f"❌ 请求体中的 JSON 无效: {e}")
            raise HTTPException(status_code=400, detail=f"请求体中的 JSON 无效: {str(e)}")
        except Exception as e:
            debug_print(f"❌ 读取请求体失败: {e}")
            raise HTTPException(status_code=400, detail=f"读取请求体失败: {str(e)}")
        
        debug_print(f"📥 请求体键: {list(body.keys())}")
        
        # 验证必填字段
        model_public_name = body.get("model")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        
        debug_print(f"🌊 流模式: {stream}")
        debug_print(f"🤖 请求的模型: {model_public_name}")
        debug_print(f"💬 消息数量: {len(messages)}")
        
        if not model_public_name:
            debug_print("❌ 请求中缺少 'model'")
            raise HTTPException(status_code=400, detail="请求体中缺少 'model'。")
        
        if not messages:
            debug_print("❌ 请求中缺少 'messages'")
            raise HTTPException(status_code=400, detail="请求体中缺少 'messages'。")
        
        if not isinstance(messages, list):
            debug_print("❌ 'messages' 必须是数组")
            raise HTTPException(status_code=400, detail="'messages' 必须是数组。")
        
        if len(messages) == 0:
            debug_print("❌ 'messages' 数组为空")
            raise HTTPException(status_code=400, detail="'messages' 数组不能为空。")

        # 从公共名称查找模型 ID
        try:
            models = get_models()
            debug_print(f"📚 已加载模型总数: {len(models)}")
        except Exception as e:
            debug_print(f"❌ 加载模型失败: {e}")
            raise HTTPException(
                status_code=503,
                detail="从 LMArena 加载模型列表失败。请稍后再试。"
            )
        
        model_id = None
        model_org = None
        model_capabilities = {}
        
        for m in models:
            if m.get("publicName") == model_public_name:
                model_id = m.get("id")
                model_org = m.get("organization")
                model_capabilities = m.get("capabilities", {})
                break
        
        if not model_id:
            debug_print(f"❌ 模型列表未找到模型 '{model_public_name}'")
            raise HTTPException(
                status_code=404, 
                detail=f"未找到模型 '{model_public_name}'。使用 /api/v1/models 查看可用模型。"
            )
        
        # 检查模型是否为隐形模型（无组织）
        if not model_org:
            debug_print(f"❌ 模型 '{model_public_name}' 是隐形模型（无组织）")
            raise HTTPException(
                status_code=403,
                detail="您无权访问隐形模型。请联系 cloudwaddie 获取更多信息。"
            )
        
        debug_print(f"✅ 找到模型 ID: {model_id}")
        debug_print(f"🔧 模型能力: {model_capabilities}")
        
        # 根据模型能力确定模态
        # 优先级: image > search > chat
        if model_capabilities.get('outputCapabilities', {}).get('image'):
            modality = "image"
        elif model_capabilities.get('outputCapabilities', {}).get('search'):
            modality = "search"
        else:
            modality = "chat"
        debug_print(f"🔍 模型模态: {modality}")

        # 记录使用情况
        try:
            model_usage_stats[model_public_name] += 1
            # 增加后立即保存统计数据
            config = get_config()
            config["usage_stats"] = dict(model_usage_stats)
            save_config(config)
        except Exception as e:
            # 如果使用情况记录失败，不要使请求失败
            debug_print(f"⚠️  记录使用统计失败: {e}")

        # 如果存在系统提示，则提取并添加到第一条用户消息之前
        system_prompt = ""
        system_messages = [m for m in messages if m.get("role") == "system"]
        if system_messages:
            system_prompt = "\n\n".join([m.get("content", "") for m in system_messages])
            debug_print(f"📋 发现系统提示: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"📋 系统提示: {system_prompt}")
        
        # 处理最后一条消息内容（可能包含图片）
        try:
            last_message_content = messages[-1].get("content", "")
            prompt, experimental_attachments = await process_message_content(last_message_content, model_capabilities)
            
            # 如果有系统提示且这是第一条用户消息，则将其添加到前面
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
                debug_print(f"✅ 系统提示已添加到用户消息前")
        except Exception as e:
            debug_print(f"❌ 处理消息内容失败: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"处理消息内容失败: {str(e)}"
            )
        
        # 验证提示
        if not prompt:
            # 如果没有文本但有附件，对于视觉模型是可以的
            if not experimental_attachments:
                debug_print("❌ 最后一条消息没有内容")
                raise HTTPException(status_code=400, detail="最后一条消息必须有内容。")
        
        # 记录提示长度以调试字符限制问题
        debug_print(f"📝 用户提示长度: {len(prompt)} 字符")
        debug_print(f"🖼️  附件: {len(experimental_attachments)} 张图片")
        debug_print(f"📝 用户提示预览: {prompt[:100]}..." if len(prompt) > 100 else f"📝 用户提示: {prompt}")
        
        # 检查合理的字符限制 (LMArena 似乎有限制)
        # 根据测试，典型限制似乎在 32K-64K 字符左右
        MAX_PROMPT_LENGTH = 113567  # 用户硬编码限制
        if len(prompt) > MAX_PROMPT_LENGTH:
            error_msg = f"提示太长 ({len(prompt)} 字符)。LMArena 的字符限制约为 {MAX_PROMPT_LENGTH} 字符。请减小消息大小。"
            debug_print(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # 使用 API 密钥 + 对话跟踪
        api_key_str = api_key["key"]
        
        # 从上下文生成对话 ID (API 密钥 + 模型 + 第一条用户消息)
        # 这允许自动会话延续而无需客户端修改
        import hashlib
        first_user_message = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        if isinstance(first_user_message, list):
            # 处理数组内容格式
            first_user_message = str(first_user_message)
        conversation_key = f"{api_key_str}_{model_public_name}_{first_user_message[:100]}"
        conversation_id = hashlib.sha256(conversation_key.encode()).hexdigest()[:16]
        
        debug_print(f"🔑 API 密钥: {api_key_str[:20]}...")
        debug_print(f"💭 自动生成的对话 ID: {conversation_id}")
        debug_print(f"🔑 对话密钥: {conversation_key[:100]}...")
        
        headers = get_request_headers()
        debug_print(f"📋 标头已准备 (认证令牌长度: {len(headers.get('Cookie', '').split('arena-auth-prod-v1=')[-1].split(';')[0])} 字符)")
        
        # 检查此 API 密钥是否存在对话
        session = chat_sessions[api_key_str].get(conversation_id)
        
        # 检测重试：如果会话存在且最后一条消息是相同的用户消息（之后没有助手响应）
        is_retry = False
        retry_message_id = None
        
        if session and len(session.get("messages", [])) >= 2:
            stored_messages = session["messages"]
            # 检查最后存储的消息是否来自具有相同内容的用户
            if stored_messages[-1]["role"] == "user" and stored_messages[-1]["content"] == prompt:
                # 这是一个重试 - 客户端再次发送相同的消息而没有助手响应
                is_retry = True
                retry_message_id = stored_messages[-1]["id"]
                # 获取需要重新生成的助手消息 ID
                if len(stored_messages) >= 2 and stored_messages[-2]["role"] == "assistant":
                    # 之前有助手响应 - 我们将重试该响应
                    retry_message_id = stored_messages[-2]["id"]
                    debug_print(f"🔁 检测到重试 - 正在重新生成助手消息 {retry_message_id}")
        
        if is_retry and retry_message_id:
            debug_print(f"🔁 使用重试端点")
            # 使用 LMArena 的重试端点
            # 格式: PUT /nextjs-api/stream/retry-evaluation-session-message/{sessionId}/messages/{messageId}
            # 注意: 我们不需要重试的有效负载，只需要 recaptchaV3Token (可选)
            payload = {
                "recaptchaV3Token": ""  # 可选，可以为空
            }
            url = f"https://lmarena.ai/nextjs-api/stream/retry-evaluation-session-message/{session['conversation_id']}/messages/{retry_message_id}"
            debug_print(f"📤 目标 URL: {url}")
            debug_print(f"📦 使用 PUT 方法重试")
            http_method = "PUT"
        elif not session:
            debug_print("🆕 创建新对话会话")
            # 新对话 - 一次生成所有 ID (就像浏览器所做的那样)
            session_id = str(uuid7())
            user_msg_id = str(uuid7())
            model_msg_id = str(uuid7())
            
            debug_print(f"🔑 生成的 session_id: {session_id}")
            debug_print(f"👤 生成的 user_msg_id: {user_msg_id}")
            debug_print(f"🤖 生成的 model_msg_id: {model_msg_id}")
            
            payload = {
                "id": session_id,
                "mode": "direct",
                "modelAId": model_id,
                "userMessageId": user_msg_id,
                "modelAMessageId": model_msg_id,
                "userMessage": {
                    "content": prompt,
                    "experimental_attachments": experimental_attachments
                },
                "modality": modality
            }
            url = "https://lmarena.ai/nextjs-api/stream/create-evaluation"
            debug_print(f"📤 目标 URL: {url}")
            debug_print(f"📦 有效负载结构: 简单的 userMessage 格式")
            debug_print(f"🔍 完整有效负载: {json.dumps(payload, indent=2)}")
            http_method = "POST"
        else:
            debug_print("🔄 使用现有对话会话")
            # 后续消息 - 生成新消息 ID
            user_msg_id = str(uuid7())
            debug_print(f"👤 生成的后续 user_msg_id: {user_msg_id}")
            model_msg_id = str(uuid7())
            debug_print(f"🤖 生成的后续 model_msg_id: {model_msg_id}")
            
            payload = {
                "id": session["conversation_id"],
                "mode": "direct",
                "modelAId": model_id,
                "userMessageId": user_msg_id,
                "modelAMessageId": model_msg_id,
                "userMessage": {
                    "content": prompt,
                    "experimental_attachments": experimental_attachments
                },
                "modality": modality
            }
            url = f"https://lmarena.ai/nextjs-api/stream/post-to-evaluation/{session['conversation_id']}"
            debug_print(f"📤 目标 URL: {url}")
            debug_print(f"📦 有效负载结构: 简单的 userMessage 格式")
            debug_print(f"🔍 完整有效负载: {json.dumps(payload, indent=2)}")
            http_method = "POST"

        debug_print(f"\n🚀 正在向 LMArena 发送 API 请求...")
        debug_print(f"⏱️  超时设置为: 120 秒")
        
        # 处理流式传输模式
        if stream:
            async def generate_stream():
                response_text = ""
                reasoning_text = ""
                citations = []
                chunk_id = f"chatcmpl-{uuid.uuid4()}"
                
                async with httpx.AsyncClient() as client:
                    try:
                        debug_print("📡 正在发送流式 POST 请求...")
                        async with client.stream('POST', url, json=payload, headers=headers, timeout=120) as response:
                            debug_print(f"✅ 流已打开 - 状态: {response.status_code}")
                            response.raise_for_status()
                            
                            async for line in response.aiter_lines():
                                line = line.strip()
                                if not line:
                                    continue
                                
                                # Parse thinking/reasoning chunks: ag:"thinking text"
                                if line.startswith("ag:"):
                                    chunk_data = line[3:]
                                    try:
                                        reasoning_chunk = json.loads(chunk_data)
                                        reasoning_text += reasoning_chunk
                                        
                                        # Send SSE-formatted chunk with reasoning_content
                                        chunk_response = {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_public_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "reasoning_content": reasoning_chunk
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_response)}\n\n"
                                        
                                    except json.JSONDecodeError:
                                        continue
                                
                                # Parse text chunks: a0:"Hello "
                                elif line.startswith("a0:"):
                                    chunk_data = line[3:]
                                    try:
                                        text_chunk = json.loads(chunk_data)
                                        response_text += text_chunk
                                        
                                        # Send SSE-formatted chunk
                                        chunk_response = {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_public_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "content": text_chunk
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_response)}\n\n"
                                        
                                    except json.JSONDecodeError:
                                        continue
                                
                                # Parse image generation: a2:[{...}] (for image models)
                                elif line.startswith("a2:"):
                                    image_data = line[3:]
                                    try:
                                        image_list = json.loads(image_data)
                                        # OpenAI format: return URL in content
                                        if isinstance(image_list, list) and len(image_list) > 0:
                                            image_obj = image_list[0]
                                            if image_obj.get('type') == 'image':
                                                image_url = image_obj.get('image', '')
                                                # Store image URL as response text for now
                                                # Will format properly in final response
                                                response_text = image_url
                                                debug_print(f"  🖼️  收到图片 URL: {image_url[:100]}...")
                                    except json.JSONDecodeError:
                                        pass
                                
                                # Parse citations/tool calls: ac:{...} (for search models)
                                elif line.startswith("ac:"):
                                    citation_data = line[3:]
                                    try:
                                        citation_obj = json.loads(citation_data)
                                        # Extract source information from argsTextDelta
                                        if 'argsTextDelta' in citation_obj:
                                            args_data = json.loads(citation_obj['argsTextDelta'])
                                            if 'source' in args_data:
                                                source = args_data['source']
                                                # Can be a single source or array of sources
                                                if isinstance(source, list):
                                                    citations.extend(source)
                                                elif isinstance(source, dict):
                                                    citations.append(source)
                                        debug_print(f"  🔗 已添加引用: {citation_obj.get('toolCallId')}")
                                    except json.JSONDecodeError:
                                        pass
                                
                                # Parse error messages
                                elif line.startswith("a3:"):
                                    error_data = line[3:]
                                    try:
                                        error_message = json.loads(error_data)
                                        print(f"  ❌ 流中出错: {error_message}")
                                    except json.JSONDecodeError:
                                        pass
                                
                                # Parse metadata for finish
                                elif line.startswith("ad:"):
                                    metadata_data = line[3:]
                                    try:
                                        metadata = json.loads(metadata_data)
                                        finish_reason = metadata.get("finishReason", "stop")
                                        
                                        # Send final chunk with finish_reason
                                        final_chunk = {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_public_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": finish_reason
                                            }]
                                        }
                                        yield f"data: {json.dumps(final_chunk)}\n\n"
                                    except json.JSONDecodeError:
                                        continue
                            
                            # Update session - Store message history with IDs (including reasoning and citations if present)
                            assistant_message = {
                                "id": model_msg_id, 
                                "role": "assistant", 
                                "content": response_text.strip()
                            }
                            if reasoning_text:
                                assistant_message["reasoning_content"] = reasoning_text.strip()
                            if citations:
                                # Deduplicate citations by URL
                                unique_citations = []
                                seen_urls = set()
                                for citation in citations:
                                    citation_url = citation.get('url')
                                    if citation_url and citation_url not in seen_urls:
                                        seen_urls.add(citation_url)
                                        unique_citations.append(citation)
                                assistant_message["citations"] = unique_citations
                            
                            if not session:
                                chat_sessions[api_key_str][conversation_id] = {
                                    "conversation_id": session_id,
                                    "model": model_public_name,
                                    "messages": [
                                        {"id": user_msg_id, "role": "user", "content": prompt},
                                        assistant_message
                                    ]
                                }
                                debug_print(f"💾 已保存对话 {conversation_id} 的新会话")
                            else:
                                # Append new messages to history
                                chat_sessions[api_key_str][conversation_id]["messages"].append(
                                    {"id": user_msg_id, "role": "user", "content": prompt}
                                )
                                chat_sessions[api_key_str][conversation_id]["messages"].append(
                                    assistant_message
                                )
                                debug_print(f"💾 已更新对话 {conversation_id} 的现有会话")
                            
                            yield "data: [DONE]\n\n"
                            debug_print(f"✅ 流已完成 - 已发送 {len(response_text)} 字符")
                            
                    except httpx.HTTPStatusError as e:
                        # Provide user-friendly error messages
                        if e.response.status_code == 429:
                            error_msg = "LMArena 超出速率限制。请稍后再试。"
                            error_type = "rate_limit_error"
                        elif e.response.status_code == 401:
                            error_msg = "未授权: 您的 LMArena 认证令牌已过期或无效。请从仪表板获取新的认证令牌。"
                            error_type = "authentication_error"
                        else:
                            error_msg = f"LMArena API 错误: {e.response.status_code}"
                            error_type = "api_error"
                        
                        print(f"❌ {error_msg}")
                        error_chunk = {
                            "error": {
                                "message": error_msg,
                                "type": error_type,
                                "code": e.response.status_code
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                    except Exception as e:
                        print(f"❌ 流错误: {str(e)}")
                        error_chunk = {
                            "error": {
                                "message": str(e),
                                "type": "internal_error"
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # Handle non-streaming mode
        async with httpx.AsyncClient() as client:
            try:
                debug_print(f"📡 正在发送 {http_method} 请求...")
                if http_method == "PUT":
                    response = await client.put(url, json=payload, headers=headers, timeout=120)
                else:
                    response = await client.post(url, json=payload, headers=headers, timeout=120)
                
                debug_print(f"✅ 收到响应 - 状态: {response.status_code}")
                debug_print(f"📏 响应长度: {len(response.text)} 字符")
                debug_print(f"📋 响应标头: {dict(response.headers)}")
                
                response.raise_for_status()
                
                debug_print(f"🔍 正在处理响应...")
                debug_print(f"📄 响应的前 500 个字符:\n{response.text[:500]}")
                
                # Process response in lmarena format
                # Format: ag:"thinking" for reasoning, a0:"text chunk" for content, ac:{...} for citations, ad:{...} for metadata
                response_text = ""
                reasoning_text = ""
                citations = []
                finish_reason = None
                line_count = 0
                text_chunks_found = 0
                reasoning_chunks_found = 0
                citation_chunks_found = 0
                metadata_found = 0
                
                debug_print(f"📊 正在解析响应行...")
                
                error_message = None
                for line in response.text.splitlines():
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse thinking/reasoning chunks: ag:"thinking text"
                    if line.startswith("ag:"):
                        chunk_data = line[3:]  # Remove "ag:" prefix
                        reasoning_chunks_found += 1
                        try:
                            # Parse as JSON string (includes quotes)
                            reasoning_chunk = json.loads(chunk_data)
                            reasoning_text += reasoning_chunk
                            if reasoning_chunks_found <= 3:  # Log first 3 reasoning chunks
                                debug_print(f"  🧠 推理块 {reasoning_chunks_found}: {repr(reasoning_chunk[:50])}")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的推理块失败: {chunk_data[:100]} - {e}")
                            continue
                    
                    # Parse text chunks: a0:"Hello "
                    elif line.startswith("a0:"):
                        chunk_data = line[3:]  # Remove "a0:" prefix
                        text_chunks_found += 1
                        try:
                            # Parse as JSON string (includes quotes)
                            text_chunk = json.loads(chunk_data)
                            response_text += text_chunk
                            if text_chunks_found <= 3:  # Log first 3 chunks
                                debug_print(f"  ✅ 块 {text_chunks_found}: {repr(text_chunk[:50])}")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的文本块失败: {chunk_data[:100]} - {e}")
                            continue
                    
                    # Parse image generation: a2:[{...}] (for image models)
                    elif line.startswith("a2:"):
                        image_data = line[3:]  # Remove "a2:" prefix
                        try:
                            image_list = json.loads(image_data)
                            # OpenAI format expects URL in content
                            if isinstance(image_list, list) and len(image_list) > 0:
                                image_obj = image_list[0]
                                if image_obj.get('type') == 'image':
                                    image_url = image_obj.get('image', '')
                                    # For image models, the URL IS the response
                                    response_text = image_url
                                    debug_print(f"  🖼️  图片 URL: {image_url[:100]}...")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的图片数据失败: {image_data[:100]} - {e}")
                            continue
                    
                    # Parse citations/tool calls: ac:{...} (for search models)
                    elif line.startswith("ac:"):
                        citation_data = line[3:]  # Remove "ac:" prefix
                        citation_chunks_found += 1
                        try:
                            citation_obj = json.loads(citation_data)
                            # Extract source information from argsTextDelta
                            if 'argsTextDelta' in citation_obj:
                                args_data = json.loads(citation_obj['argsTextDelta'])
                                if 'source' in args_data:
                                    source = args_data['source']
                                    # Can be a single source or array of sources
                                    if isinstance(source, list):
                                        citations.extend(source)
                                    elif isinstance(source, dict):
                                        citations.append(source)
                            if citation_chunks_found <= 3:  # Log first 3 citations
                                debug_print(f"  🔗 引用块 {citation_chunks_found}: {citation_obj.get('toolCallId')}")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的引用块失败: {citation_data[:100]} - {e}")
                            continue
                    
                    # Parse error messages: a3:"An error occurred"
                    elif line.startswith("a3:"):
                        error_data = line[3:]  # Remove "a3:" prefix
                        try:
                            error_message = json.loads(error_data)
                            debug_print(f"  ❌ 收到错误消息: {error_message}")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的错误消息失败: {error_data[:100]} - {e}")
                            error_message = error_data
                    
                    # Parse metadata: ad:{"finishReason":"stop"}
                    elif line.startswith("ad:"):
                        metadata_data = line[3:]  # Remove "ad:" prefix
                        metadata_found += 1
                        try:
                            metadata = json.loads(metadata_data)
                            finish_reason = metadata.get("finishReason")
                            debug_print(f"  📋 发现元数据: finishReason={finish_reason}")
                        except json.JSONDecodeError as e:
                            debug_print(f"  ⚠️ 解析第 {line_count} 行的元数据失败: {metadata_data[:100]} - {e}")
                            continue
                    elif line.strip():  # Non-empty line that doesn't match expected format
                        if line_count <= 5:  # Log first 5 unexpected lines
                            debug_print(f"  ❓ 意外的行格式 {line_count}: {line[:100]}")

                debug_print(f"\n📊 解析摘要:")
                debug_print(f"  - 总行数: {line_count}")
                debug_print(f"  - 发现推理块: {reasoning_chunks_found}")
                debug_print(f"  - 发现文本块: {text_chunks_found}")
                debug_print(f"  - 发现引用块: {citation_chunks_found}")
                debug_print(f"  - 元数据条目: {metadata_found}")
                debug_print(f"  - 最终响应长度: {len(response_text)} 字符")
                debug_print(f"  - 最终推理长度: {len(reasoning_text)} 字符")
                debug_print(f"  - 发现引用: {len(citations)}")
                debug_print(f"  - 完成原因: {finish_reason}")
                
                if not response_text:
                    debug_print(f"\n⚠️  警告: 响应文本为空!")
                    debug_print(f"📄 完整原始响应:\n{response.text}")
                    if error_message:
                        error_detail = f"LMArena API 错误: {error_message}"
                        print(f"❌ {error_detail}")
                        # Return OpenAI-compatible error response
                        return {
                            "error": {
                                "message": error_detail,
                                "type": "upstream_error",
                                "code": "lmarena_error"
                            }
                        }
                    else:
                        error_detail = "LMArena API 返回空响应。这可能是由于: 无效的认证令牌、cf_clearance 过期、模型不可用或 API 速率限制。"
                        debug_print(f"❌ {error_detail}")
                        # Return OpenAI-compatible error response
                        return {
                            "error": {
                                "message": error_detail,
                                "type": "upstream_error",
                                "code": "empty_response"
                            }
                        }
                else:
                    debug_print(f"✅ 响应文本预览: {response_text[:200]}...")
                
                # Update session - Store message history with IDs (including reasoning and citations if present)
                assistant_message = {
                    "id": model_msg_id, 
                    "role": "assistant", 
                    "content": response_text.strip()
                }
                if reasoning_text:
                    assistant_message["reasoning_content"] = reasoning_text.strip()
                if citations:
                    # Deduplicate citations by URL
                    unique_citations = []
                    seen_urls = set()
                    for citation in citations:
                        citation_url = citation.get('url')
                        if citation_url and citation_url not in seen_urls:
                            seen_urls.add(citation_url)
                            unique_citations.append(citation)
                    assistant_message["citations"] = unique_citations
                
                if not session:
                    chat_sessions[api_key_str][conversation_id] = {
                        "conversation_id": session_id,
                        "model": model_public_name,
                        "messages": [
                            {"id": user_msg_id, "role": "user", "content": prompt},
                            assistant_message
                        ]
                    }
                    debug_print(f"💾 已保存对话 {conversation_id} 的新会话")
                else:
                    # Append new messages to history
                    chat_sessions[api_key_str][conversation_id]["messages"].append(
                        {"id": user_msg_id, "role": "user", "content": prompt}
                    )
                    chat_sessions[api_key_str][conversation_id]["messages"].append(
                        assistant_message
                    )
                    debug_print(f"💾 已更新对话 {conversation_id} 的现有会话")

                # Build message object with reasoning and citations if present
                message_obj = {
                    "role": "assistant",
                    "content": response_text.strip(),
                }
                if reasoning_text:
                    message_obj["reasoning_content"] = reasoning_text.strip()
                if citations:
                    # Deduplicate citations by URL
                    unique_citations = []
                    seen_urls = set()
                    for citation in citations:
                        citation_url = citation.get('url')
                        if citation_url and citation_url not in seen_urls:
                            seen_urls.add(citation_url)
                            unique_citations.append(citation)
                    message_obj["citations"] = unique_citations
                
                # Calculate token counts (including reasoning tokens)
                prompt_tokens = len(prompt)
                completion_tokens = len(response_text)
                reasoning_tokens = len(reasoning_text)
                total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
                
                # Build usage object with reasoning tokens if present
                usage_obj = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                if reasoning_tokens > 0:
                    usage_obj["reasoning_tokens"] = reasoning_tokens
                
                final_response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_public_name,
                    "conversation_id": conversation_id,
                    "choices": [{
                        "index": 0,
                        "message": message_obj,
                        "finish_reason": "stop"
                    }],
                    "usage": usage_obj
                }
                
                debug_print(f"\n✅ 请求成功完成")
                debug_print("="*80 + "\n")
                
                return final_response

            except httpx.HTTPStatusError as e:
                # Provide user-friendly error messages
                if e.response.status_code == 429:
                    error_detail = "LMArena 超出速率限制。请稍后再试。"
                    error_type = "rate_limit_error"
                elif e.response.status_code == 401:
                    error_detail = "未授权: 您的 LMArena 认证令牌已过期或无效。请从仪表板获取新的认证令牌。"
                    error_type = "authentication_error"
                else:
                    error_detail = f"LMArena API 错误: {e.response.status_code}"
                    try:
                        error_body = e.response.json()
                        error_detail += f" - {error_body}"
                    except:
                        error_detail += f" - {e.response.text[:200]}"
                    error_type = "upstream_error"
                
                print(f"\n❌ HTTP 状态错误")
                print(f"📛 错误详情: {error_detail}")
                print(f"📤 请求 URL: {url}")
                debug_print(f"📤 请求有效负载 (已截断): {json.dumps(payload, indent=2)[:500]}")
                debug_print(f"📥 响应文本: {e.response.text[:500]}")
                print("="*80 + "\n")
                
                # Return OpenAI-compatible error response
                return {
                    "error": {
                        "message": error_detail,
                        "type": error_type,
                        "code": f"http_{e.response.status_code}"
                    }
                }
            
            except httpx.TimeoutException as e:
                print(f"\n⏱️  超时错误")
                print(f"📛 请求在 120 秒后超时")
                print(f"📤 请求 URL: {url}")
                print("="*80 + "\n")
                # Return OpenAI-compatible error response
                return {
                    "error": {
                        "message": "LMArena API 请求在 120 秒后超时",
                        "type": "timeout_error",
                        "code": "request_timeout"
                    }
                }
            
            except Exception as e:
                print(f"\n❌ HTTP 客户端发生意外错误")
                print(f"📛 错误类型: {type(e).__name__}")
                print(f"📛 错误消息: {str(e)}")
                print(f"📤 请求 URL: {url}")
                print("="*80 + "\n")
                # Return OpenAI-compatible error response
                return {
                    "error": {
                        "message": f"意外错误: {str(e)}",
                        "type": "internal_error",
                        "code": type(e).__name__.lower()
                    }
                }
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ 顶级异常")
        print(f"📛 错误类型: {type(e).__name__}")
        print(f"📛 错误消息: {str(e)}")
        print("="*80 + "\n")
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LMArena Bridge 服务器正在启动...")
    print("=" * 60)
    print(f"📍 仪表板: http://localhost:{PORT}/dashboard")
    print(f"🔐 登录: http://localhost:{PORT}/login")
    print(f"📚 API 基础 URL: http://localhost:{PORT}/api/v1")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT)