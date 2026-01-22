## 1. 查找当前文件夹的wav文件（迭代子文件夹）
    find ./ -type f \( -iname "*.wav" \) | wc -l
    find ./ -name "*.wav" | wc -l


    