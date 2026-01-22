# pdb进行debug
import pdb;pdb.set_trace()


# Vscode进行debug
## 用法：
    python -m debugpy --listen 2221 --wait-for-client cosyvoice_cfm.py

## VScode配置：
    {
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
    
            {
                "name": "测试连接",
                "type": "python",
                "request": "attach",
                "connect": {
                    "host": "127.0.0.1",
                    "port": 6666
                },
                "justMyCode": true
            }
    
        ]
    }

