#!/bin/bash

# RAG System with BiliGo - 使用 tmux 启动多个服务
# 用法: bash start_services.sh

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SESSION_NAME="rag-system"

echo "📦 启动 RAG System 服务..."
echo "项目目录: $PROJECT_DIR"

# 杀死已存在的 session
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# 创建新的 tmux session
tmux new-session -d -s $SESSION_NAME -x 200 -y 50

# 第 1 个窗口：RAG API 服务
echo "🚀 启动 RAG API 服务 (端口 8000)..."
tmux new-window -t $SESSION_NAME -n "rag-api"
tmux send-keys -t $SESSION_NAME:rag-api "cd $PROJECT_DIR && python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000" C-m

# 第 2 个窗口：BiliGo Flask 应用
echo "🤖 启动 BiliGo 应用 (端口 4999)..."
tmux new-window -t $SESSION_NAME -n "biligo"
tmux send-keys -t $SESSION_NAME:biligo "cd $PROJECT_DIR/BiliGo && python3 app.py" C-m

# 第 3 个窗口：日志/命令窗口（默认选中）
echo "📊 创建日志窗口..."
tmux new-window -t $SESSION_NAME -n "logs"
tmux send-keys -t $SESSION_NAME:logs "cd $PROJECT_DIR && echo '📋 RAG System 已启动!' && echo '' && echo '服务状态:' && echo '  RAG API: http://localhost:8000/health' && echo '  BiliGo:  http://localhost:4999' && echo '' && echo '常用命令:' && echo '  查看所有窗口: tmux list-windows -t $SESSION_NAME' && echo '  切换窗口: tmux select-window -t $SESSION_NAME:rag-api' && echo '  查看 RAG 日志: tmux capture-pane -t $SESSION_NAME:rag-api -p' && echo '  查看 BiliGo 日志: tmux capture-pane -t $SESSION_NAME:biligo -p' && echo '  停止所有服务: tmux kill-session -t $SESSION_NAME'" C-m

# 选中日志窗口
tmux select-window -t $SESSION_NAME:logs

echo ""
echo "✅ 所有服务已启动！"
echo ""
echo "📌 tmux 命令:"
echo "  进入 session:          tmux attach -t $SESSION_NAME"
echo "  退出 session:          按 Ctrl+B 再按 D"
echo "  查看所有窗口:          tmux list-windows -t $SESSION_NAME"
echo "  切换到 rag-api:        tmux select-window -t $SESSION_NAME:rag-api"
echo "  切换到 biligo:         tmux select-window -t $SESSION_NAME:biligo"
echo "  查看 rag-api 日志:      tmux capture-pane -t $SESSION_NAME:rag-api -p"
echo "  查看 biligo 日志:       tmux capture-pane -t $SESSION_NAME:biligo -p"
echo "  停止所有服务:          tmux kill-session -t $SESSION_NAME"
echo ""
echo "🌐 Web 访问:"
echo "  RAG API 健康检查:      curl http://localhost:8000/health"
echo "  BiliGo 配置界面:      http://localhost:4999"
echo ""
