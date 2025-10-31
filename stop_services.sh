#!/bin/bash

# 停止 RAG System 服务
SESSION_NAME="rag-system"

echo "🛑 停止 RAG System 服务..."

if tmux list-sessions 2>/dev/null | grep -q "^$SESSION_NAME"; then
    tmux kill-session -t $SESSION_NAME
    echo "✅ 服务已停止"
else
    echo "⚠️ 没有找到 '$SESSION_NAME' session"
fi
